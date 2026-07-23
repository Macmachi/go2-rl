import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False,
                 extra_morphs=None, camera_cfg=None, extra_sensors=None):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                # auto-collision indispensable pour les postures bipèdes : sans
                # elle les pattes se traversent (constaté sur go2-step v1)
                enable_self_collision=bool(env_cfg.get("self_collision", False)),
                tolerance=1e-5,
                max_collision_pairs=80 if env_cfg.get("self_collision") else 20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            # env_cfg["vis_options"] : éclairage/fond personnalisés (ex. DAYLIGHT de
            # city_scene pour la démo ville) — défaut Genesis sinon.
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0],
                                              **env_cfg.get("vis_options", {})),
            show_viewer=show_viewer,
        )

        # sol : terrain à relief (si terrain_cfg fourni) OU plateau plat (défaut).
        # Le terrain sert à la locomotion tout-usage ; le plat reste le défaut
        # pour toutes les policies existantes (rétro-compatible).
        self.terrain = None
        terrain_cfg = self.env_cfg.get("terrain_cfg")
        if terrain_cfg:
            self.terrain = self.scene.add_entity(gs.morphs.Terrain(**terrain_cfg))
        else:
            self.scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/plane/plane.urdf",
                    fixed=True,
                )
            )

        # add robot — links_to_keep préserve les liens de PIEDS (sinon fusionnés
        # dans les tibias : impossible de distinguer appui pied / appui jarret).
        # robot_urdf permet la variante avec charge caméra dorsale.
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.env_cfg.get("robot_urdf", "urdf/go2/urdf/go2.urdf"),
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
                links_to_keep=self.env_cfg.get("links_to_keep", []),
            ),
        )

        # optional extra entities (props for scenarios, e.g. a bag to inspect)
        self.extra_entities = []
        if extra_morphs:
            for morph in extra_morphs:
                # un item peut être (morph, surface) — ex. spots émissifs du décor ville
                if isinstance(morph, tuple):
                    self.extra_entities.append(self.scene.add_entity(morph[0], surface=morph[1]))
                else:
                    self.extra_entities.append(self.scene.add_entity(morph))

        # optional offscreen camera(s) (embedded robot view / video recording)
        # camera_cfg : dict (une caméra) ou liste de dicts (plusieurs)
        self.cams = []
        self.cam = None
        if camera_cfg is not None:
            cfgs = camera_cfg if isinstance(camera_cfg, (list, tuple)) else [camera_cfg]
            for c in cfgs:
                self.cams.append(self.scene.add_camera(
                    res=c.get("res", (640, 480)),
                    pos=c.get("pos", (2.0, 0.0, 1.0)),
                    lookat=c.get("lookat", (0.0, 0.0, 0.3)),
                    fov=c.get("fov", 70),
                    GUI=False,
                    debug=c.get("debug", False),   # True = rend aussi les objets draw_debug_* (rayons LiDAR)
                ))
            self.cam = self.cams[0]

        # optional sensors (e.g. lidar) — added before build
        self.sensors = []
        if extra_sensors:
            for opts in extra_sensors:
                self.sensors.append(self.scene.add_sensor(opts))

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)
        # BORNAGE DU COUPLE aux specs moteur réel Unitree Go2 (GO-M8010-6) :
        # hanche/cuisse 23,7 N·m, genou 35,55 N·m (couples nominaux ; crête ~45 N·m).
        # Sans ça, le PD peut demander un couple irréaliste. Activé via
        # env_cfg["enforce_motor_limits"] (réalisme sim→réel).
        if self.env_cfg.get("enforce_motor_limits", False):
            eff = self.env_cfg.get("motor_effort_limits", [23.7, 23.7, 35.55] * 4)
            eff_t = torch.tensor(eff, dtype=gs.tc_float, device=gs.device)
            self.robot.set_dofs_force_range(-eff_t, eff_t, self.motors_dof_idx)

        # BORNAGE DES CIBLES articulaires aux limites de position URDF : le firmware
        # du vrai Go2 clippe toute consigne hors plage articulaire. Opt-in via
        # env_cfg["clamp_targets_to_limits"] (les anciennes policies, entraînées sans,
        # sont rendues à l'identique — la clé est absente de leur cfgs.pkl).
        self._target_clamp = None
        if self.env_cfg.get("clamp_targets_to_limits", False):
            lo, hi = self.robot.get_dofs_limit(self.motors_dof_idx)
            self._target_clamp = (torch.as_tensor(lo, dtype=gs.tc_float, device=gs.device),
                                  torch.as_tensor(hi, dtype=gs.tc_float, device=gs.device))

        # Define global gravity direction vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # Initial state
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        # Tous les DOF actionnés (pattes + éventuel bras), dans l'ordre du robot :
        # sert à reconstruire le qpos complet pour set_qpos au reset.
        _all_actuated = [
            (joint.name, self.env_cfg["default_joint_angles"][joint.name])
            for joint in self.robot.joints[1:] if getattr(joint, "n_dofs", 1) > 0
        ]
        _init_dof_pos_full = torch.tensor(
            [v for _, v in _all_actuated], dtype=gs.tc_float, device=gs.device
        )
        # Buffer dof_pos = uniquement les DOF MOTEURS pilotés par la policy (12 pattes).
        # Filtrer sur joint_names garde le comportement d'origine à l'identique pour
        # les robots sans bras (tous les actionnés sont des pattes) et évite le
        # mismatch de taille quand un bras articulé ajoute des DOF non pilotés ici.
        _leg_names = set(self.env_cfg["joint_names"])
        self.init_dof_pos = torch.tensor(
            [v for n, v in _all_actuated if n in _leg_names], dtype=gs.tc_float, device=gs.device
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, _init_dof_pos_full))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # initialize buffers
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        # indices des HANCHES (abduction) — pour la pénalité posture "hip_straight"
        self.hip_dof_idx = [i for i, n in enumerate(self.env_cfg["joint_names"]) if "hip" in n]
        self.extras = dict()  # extra information for logging

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        # --- RANDOMISATION DE DOMAINE (réalisme sim→réel), activée par env_cfg ---
        # friction sol variable, poussées aléatoires, charge dorsale randomisée +
        # limites de vitesse articulaire Unitree. Tout est OPT-IN (None/0 = inactif),
        # donc sans effet sur un entraînement qui ne les configure pas.
        self.n_links = len(self.robot.links)
        fr = self.env_cfg.get("friction_range")                 # ex. (0.4, 1.6)
        self._fric_lo, self._fric_hi = (float(fr[0]), float(fr[1])) if fr else (None, None)
        self._payload_kg = float(self.env_cfg.get("payload_rand_kg", 0.0))
        self._push_interval = (int(self.env_cfg["push_interval_s"] / self.dt)
                               if self.env_cfg.get("push_interval_s") else 0)
        self._push_vel = float(self.env_cfg.get("push_vel", 0.0))
        # phase de poussée PAR ENV (démarrage aléatoire) : sans ça les 4096 robots
        # seraient bousculés au même instant — perturbation synchrone, DR appauvrie.
        self._push_ctr = (torch.randint(0, max(self._push_interval, 1), (self.num_envs,),
                                        device=gs.device)
                          if self._push_interval > 0 else None)
        # vitesses articulaires nominales Unitree Go2 (hanche/cuisse 30,1 ; genou 20,07 rad/s)
        self.dof_vel_limits = torch.tensor(
            self.env_cfg.get("dof_vel_limits", [30.1, 30.1, 20.07] * 4),
            dtype=gs.tc_float, device=gs.device)
        # BRUIT DE CAPTEURS (IMU + encodeurs, jamais parfaits sur le vrai robot) :
        # bruit uniforme ±scale ajouté aux champs MESURÉS de l'observation. Les
        # commandes et les actions passées sont internes au contrôleur → pas de bruit.
        # Échelles exprimées en unités physiques, converties en unités d'obs.
        noise = self.env_cfg.get("obs_noise")
        if noise:
            self.obs_noise_vec = torch.concatenate((
                torch.full((3,), float(noise.get("ang_vel", 0.0)) * self.obs_scales["ang_vel"]),
                torch.full((3,), float(noise.get("gravity", 0.0))),
                torch.zeros(self.num_commands),
                torch.full((self.num_actions,), float(noise.get("dof_pos", 0.0)) * self.obs_scales["dof_pos"]),
                torch.full((self.num_actions,), float(noise.get("dof_vel", 0.0)) * self.obs_scales["dof_vel"]),
                torch.zeros(self.num_actions),
            )).to(dtype=gs.tc_float, device=gs.device)
        else:
            self.obs_noise_vec = None
        # indices des PIEDS (foulée) et des liens à contact INDÉSIRABLE (corps/cuisses
        # au sol = trébuche). Les liens de pieds n'existent séparément que si
        # links_to_keep les préserve (sinon fusionnés dans les tibias) → garde robuste.
        _names = [l.name for l in self.robot.links]
        _foot = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        self.feet_idx = [_names.index(n) for n in _foot if n in _names]
        self.penal_idx = [i for i, n in enumerate(_names)
                          if ("thigh" in n or "hip" in n or n == "base")]
        self.feet_air_time = (torch.zeros((self.num_envs, len(self.feet_idx)),
                                          dtype=gs.tc_float, device=gs.device)
                              if self.feet_idx else None)
        self._dr_ready = True

        self.reset()

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        # Une FRACTION des envs reçoit la commande NULLE : le robot apprend à
        # S'IMMOBILISER franchement. Sans ça, une loi continue ne tombe jamais pile
        # à (0,0,0) → il n'entraîne jamais l'arrêt et dérive/piétine quand on relâche
        # les commandes. Activé via env_cfg["zero_cmd_frac"] (0 = comportement d'avant).
        frac = self.env_cfg.get("zero_cmd_frac", 0.0)
        if frac > 0.0:
            zero = torch.rand(self.num_envs, device=gs.device) < frac
            commands = torch.where(zero[:, None], torch.zeros_like(commands), commands)
        # CURRICULUM DE VITESSE (course) : le plafond de vx AVANT commandé monte
        # graduellement de vx_cap_start -> vx_cap_end sur vx_ramp_steps pas. Sauter
        # direct au max fait décrocher la policy (peu de gradient utile). Inactif si
        # vx_cap_start absent (rétro-compatible marche).
        cap0 = self.env_cfg.get("vx_cap_start")
        if cap0 is not None:
            self._run_step = getattr(self, "_run_step", 0) + 1
            capN = self.env_cfg.get("vx_cap_end", cap0)
            ramp = max(1, int(self.env_cfg.get("vx_ramp_steps", 1)))
            cap = cap0 + (capN - cap0) * min(1.0, self._run_step / ramp)
            commands[:, 0].clamp_(max=cap)
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        if self._target_clamp is not None:
            target_dof_pos = torch.clamp(target_dof_pos, self._target_clamp[0], self._target_clamp[1])
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # resample commands
        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # Compute timeout
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # Reset environment if necessary
        self._reset_idx(self.reset_buf)

        # update observations
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        # --- POUSSÉES aléatoires (bousculade foule/porte/rafale) : impulsion
        # horizontale sur la base → apprend à encaisser et se rattraper. Phase PAR ENV
        # (chaque robot est poussé à son propre moment, pas tous en même temps).
        if getattr(self, "_dr_ready", False) and self._push_interval > 0:
            self._push_ctr += 1
            due = self._push_ctr >= self._push_interval
            if bool(due.any()):
                self._push_ctr[due] = 0
                v = self.robot.get_dofs_velocity(dofs_idx_local=[0, 1])   # vx,vy base (monde)
                kick = ((torch.rand(self.num_envs, 2, device=gs.device) * 2 - 1)
                        * self._push_vel * due.unsqueeze(1).float())      # kick des seuls envs "dus"
                self.robot.set_dofs_velocity(v + kick, dofs_idx_local=[0, 1], skip_forward=True)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        # reset state
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # reset buffers
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        # --- RANDOMISATION DE DOMAINE au reset : friction sol + charge dorsale ---
        # (sols variés carrelage/gravier/béton + accessoires embarqués). envs_idx est
        # un MASQUE booléen (ou None) → conversion en indices pour les setters Genesis.
        if getattr(self, "_dr_ready", False):
            idx = (torch.arange(self.num_envs, device=gs.device) if envs_idx is None
                   else torch.nonzero(envs_idx, as_tuple=False).squeeze(1))
            n = int(idx.numel())
            if n > 0 and self._fric_lo is not None:
                ratio = torch.rand(n, device=gs.device) * (self._fric_hi - self._fric_lo) + self._fric_lo
                self.robot.set_friction_ratio(
                    ratio.unsqueeze(1).repeat(1, self.n_links),
                    links_idx_local=list(range(self.n_links)), envs_idx=idx)
            if n > 0 and self._payload_kg > 0.0:
                shift = torch.rand(n, 1, device=gs.device) * self._payload_kg
                self.robot.set_mass_shift(shift, links_idx_local=[0], envs_idx=idx)
            if self.feet_air_time is not None:
                self.feet_air_time[idx] = 0.0

        # random sample command upon reset
        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )
        # bruit de capteurs (réalisme sim->réel) — voir __init__ / obs_noise
        if getattr(self, "obs_noise_vec", None) is not None:
            self.obs_buf += (torch.rand_like(self.obs_buf) * 2.0 - 1.0) * self.obs_noise_vec

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalise la vitesse verticale de la base. BORNÉ (audit 22/07) : en terrain,
        # un robot warm-starté sur du plat CULBUTE parfois (v_z ~ plusieurs m/s) →
        # square non borné explosait (pic récompense −119) → gradient PPO catastrophique
        # → policy effondrée (constaté sur pente ET escalier). Le clip à 4.0 (|v_z| ~2 m/s)
        # n'affecte pas la locomotion normale (v_z faible) mais coupe le pic de divergence.
        return torch.nan_to_num(torch.square(self.base_lin_vel[:, 2]).clip(max=4.0),
                                nan=0.0, posinf=4.0, neginf=0.0)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_hip_straight(self):
        # Pénalise l'ABDUCTION des hanches (pattes écartées "en biais") : garde les
        # jambes verticales sous le corps. Actif seulement si "hip_straight" est dans
        # reward_scales (les anciennes policies ne sont pas affectées).
        dev = self.dof_pos[:, self.hip_dof_idx] - self.default_dof_pos[self.hip_dof_idx]
        return torch.sum(torch.square(dev), dim=1)

    # ---- variantes CONDITIONNÉES À LA VITESSE (posture exigée à l'arrêt/marche,
    # galop LIBRE : pénaliser hauteur/hanches à grande vitesse tue le sprint —
    # constaté sur 3 retrains course successifs le 20/07) ----
    def _reward_base_height_slow(self):
        slow = (self.commands[:, 0].abs() < 1.0).float()
        return slow * torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_hip_straight_slow(self):
        slow = (self.commands[:, 0].abs() < 1.0).float()
        dev = self.dof_pos[:, self.hip_dof_idx] - self.default_dof_pos[self.hip_dof_idx]
        return slow * torch.sum(torch.square(dev), dim=1)

    # ---- POSTURE « bien droite » (repris du marcheur robust, version sol plat) ----
    def _reward_orientation(self):
        # Tronc À PLAT : pénalise l'inclinaison du corps (composantes xy de la
        # gravité projetée dans le repère base). C'est LA clé du « bien droit » —
        # sans ce terme, un marcheur se fige souvent penché. Borné (gravité ∈ [-1,1]).
        return torch.nan_to_num(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1),
                                nan=0.0, posinf=2.0, neginf=0.0)

    def _reward_stand_still(self):
        # À commande NULLE : se figer. Pénalise le mouvement de base (translation +
        # lacet ×3, la dérive en cap étant le défaut le plus visible à l'arrêt).
        still = (torch.linalg.norm(self.commands[:, :2], dim=1) < 0.05) & \
                (self.commands[:, 2].abs() < 0.05)
        lin = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        ang = torch.square(self.base_ang_vel[:, 2])
        motion = (lin + 3.0 * ang).clip(0.0, 12.0)
        return torch.nan_to_num(motion * still.float(), nan=0.0, posinf=12.0, neginf=0.0)

    def _reward_feet_air_time(self):
        # Récompense une VRAIE foulée (pied en l'air ~0,3-0,5 s) quand le robot avance :
        # pousse à lever franchement les pattes → trot/galop avec phase de vol, plutôt
        # qu'un piétinement traînant. Actif seulement en mouvement commandé.
        if self.feet_air_time is None:
            return torch.zeros(self.num_envs, device=gs.device)
        f = self.robot.get_links_net_contact_force()
        contact = torch.linalg.norm(f[:, self.feet_idx], dim=2) > 1.0
        self.feet_air_time = (self.feet_air_time + self.dt).clip(max=2.0)
        touchdown = contact & (self.feet_air_time > 0.0)
        contrib = (self.feet_air_time - 0.3).clip(-0.3, 0.5)
        rew = torch.where(touchdown, contrib, torch.zeros_like(contrib)).sum(dim=1)
        self.feet_air_time = torch.where(contact, torch.zeros_like(self.feet_air_time),
                                         self.feet_air_time)
        moving = torch.linalg.norm(self.commands[:, :2], dim=1) > 0.1
        return torch.nan_to_num(rew * moving.float(), nan=0.0)

    def _reward_undesired_contact(self):
        # Pénalise les contacts au sol du CORPS / des CUISSES / des hanches (le robot
        # rampe ou trébuche). Compte le nombre de liens pénalisés en contact franc.
        f = self.robot.get_links_net_contact_force()
        mag = torch.linalg.norm(f[:, self.penal_idx], dim=2)
        return torch.nan_to_num((mag > 1.0).float().sum(dim=1), nan=0.0)

    def _reward_dof_vel_limit(self):
        # Pénalise le DÉPASSEMENT des vitesses articulaires nominales Unitree
        # (30,1 rad/s hanche/cuisse ; 20,07 rad/s genou) → garde les servos dans les
        # specs constructeur. Borné pour ne pas empoisonner le gradient PPO.
        over = (self.dof_vel.abs() - self.dof_vel_limits).clip(min=0.0, max=5.0)
        return torch.nan_to_num(torch.sum(torch.square(over), dim=1), nan=0.0,
                                posinf=75.0, neginf=0.0)

    def _reward_dof_freeze_stop(self):
        # À commande NULLE : les PATTES doivent cesser de bouger (vitesses
        # articulaires ~0). Complète stand_still (base immobile) pour un arrêt NET :
        # après avoir marché, le robot se pose et ne piétine plus.
        still = (torch.linalg.norm(self.commands[:, :2], dim=1) < 0.05) & \
                (self.commands[:, 2].abs() < 0.05)
        motion = torch.sum(torch.square(self.dof_vel), dim=1).clip(0.0, 60.0)
        return torch.nan_to_num(motion * still.float(), nan=0.0, posinf=60.0, neginf=0.0)
