# Simulation Parameters

## Introduction
This table defines the configuration settings for the simfish simulation environment. Note that distances are measured in pixels (the default values are based on a ratio of 10 pixels = 1 mm).

---

## 1. Global Simulation Settings
General constraints for the simulation engine and time-stepping.

| Name | Meaning | Example Value | Data Type | Optional |
| :--- | :--- | :--- | :--- | :--- |
| `test_sensory_system` | Enables diagnostic mode for visual inputs | `false` | Boolean | No |
| `max_sim_steps_per_episode` | Max steps before an episode ends | `1000` | Integer | No |
| `phys_steps_per_sim_step` | Physics model sub-steps for each simulation step | `100` | Integer | No |
| `sim_step_duration_seconds` | Real-world time delta per step in seconds | `0.2` | Float | No |

---

## 2. Arena
Defines the physical boundaries and lighting conditions.

| Name | Meaning | Example Value | Data Type | Optional |
| :--- | :--- | :--- | :--- | :--- |
| `width` / `height` | Arena dimensions (pixels) | `5000` | Integer | No |
| `dark_fraction` | Fraction of the arena that is dark | `0.0` | Float | No |
| `light_gradient` | Steepness of light-to-dark transitions (pixels) | `20` | Integer | No |
| `dark_gain` | Dark region illumination fraction | `0.1` | Float | No |
| `light_decay_rate` | Rate at which light intensity decreases over distance | `0.01` | Float | No |
| `uv_object_intensity` | Multiplier for UV-reflective objects | `1.2` | Float | No |
| `bottom_intensity` | Intensity of light scattering at the bottom of the arena | `60.0` | Float | No |
| `sediment_sigma` | Standard deviation for sediment texture | `5.0` | Float | No |
| `water_uv_scatter` | UV scattering coefficient for water | `0.1` | Float | No |
| `wall_bounce` | If true, fish reflects off walls | `false` | Boolean | No |

---

## 3. Fish
Parameters governing the physical body and the visual system of the agent.

[Image of larval zebrafish visual field and eye vergence]

| Name | Meaning | Example Value | Data Type | Optional |
| :--- | :--- | :--- | :--- | :--- |
| `mouth_radius` | Size of prey capture area (pixels) | `2.0` | Float | No |
| `head_radius` | Radius of fish head (pixels) | `2.5` | Float | No |
| `tail_length` | Physical tail length (pixels) | `41.0` | Float | No |
| `elevation` | Height of the fish's above the bottom (pixels) | `50` | Integer | No |
| `deterministic_action` | If true, fish actions are deterministic, based on mean value | `false` | Boolean | No |

---

## 4. Eyes and vision
Parameters governing the visual system of the agent.

| Name | Meaning | Example Value | Data Type | Optional |
| :--- | :--- | :--- | :--- | :--- |
| `verg_angle` | Eye vergence angle (degrees) | `77.0` | Float | No |
| `visual_field` | Total degrees of vision per eye | `163.0` | Float | No |
| `biasx` | Horizontal offset of each eye from the midline (pixels) | `2.5` | Float | No |
| `shot_noise` | If true, adds shot noise to visual input | `true` | Boolean | No |
| `uv_photoreceptor_rf_size` | Size of UV photoreceptor receptive fields (radians) | `0.0133` | Float | No |
| `red_photoreceptor_rf_size` | Size of red photoreceptor receptive fields (radians) | `0.0133` | Float | No |
| `sz_rf_spacing` | Spacing of RFs in the strike zone (radians) | `0.04` | Float | No |
| `sz_size` | Size of the strike zone (radians) | `1.047` | Float | No |
| `sz_oversampling_factor` | Oversampling factor PRs in the strike zone | `2.5` | Float | No |
| `sz_edge_steepness` | Steepness of the strike zone edge | `5.0` | Float | No |
| `viewing_elevations` | Angles of bottom viewing (degrees) | `[50, 75]` | List | No |

---

## 5. Capture swim
| Name | Meaning | Example Value | Data Type | Optional |
| :--- | :--- | :--- | :--- | :--- |
| `permissive_time_fraction` | Fraction of time within the bout that still allows capture | `0.5` | Float | No |
| `permissive_angle` | Angle within which capture is allowed (degrees) | `0.785` | Float | No |
| `energy_cost_scaling` | Multiplier for energy cost during capture | `5` | Float | No |

---

## 6. Prey
Behavioral variables for prey items (paramecia)

| Name | Meaning | Example Value | Data Type | Optional |
| :--- | :--- | :--- | :--- | :--- |
| `radius` | Collision radius of prey | `1.0` | Float | No |
| `num` | Number of prey items | `30` | Integer | No |
| `sensing_distance` | Distance at which prey can sense the fish (pixels) | `100` | Float | No |
| `max_turning_angle` | Maximum angle prey can turn (radians) | `0.15` | Float | No |
| `cloud_num` | Number of prey clouds (zero for even distribution) | `3` | Integer | No |
| `p_slow` | Probability of slow gait | `0.8` | Float | No |
| `p_fast` | Probability of fast gait | `0.2` | Float | No |
| `p_escape` | Probability of escape gait if in range| `0.3` | Float | No |
| `p_switch` | Probability of switching gait | `0.05` | Float | No |
| `p_large_turn` | Probability of large turn behavior | `0.04` | Float | No |
| `velocity_slow` | Speed during slow movement (px/sec) | `10.0` | Float | No |
| `velocity_fast` | Speed during fast movement (px/sec) | `20.0` | Float | No |
| `velocity_jump` | Speed during a prey escape burst (px/sec) | `1800` | Integer | No |
| `reproduction_mode` | If true, prey can reproduce | `false` | Boolean | No |
| `birth_rate` | Probability of new prey spawning | `0.002` | Float | No |
| `cloud_region_size` | Size of the prey clouds | `200` | Integer | No |
| `safe_duration` | Minimal prey lifespan (steps) | `100` | Integer | No |
| `p_death` | Probability of prey death | `0.001` | Float | No |

---

## 7. Predator
Behavioral variables for predator items

| Name | Meaning | Example Value | Data Type | Optional |
| :--- | :--- | :--- | :--- | :--- |
| `radius` | Hitbox/shadow radius of the predator | `64` | Integer | No |
| `velocity` | Movement speed of the predator (px/sec) | `90` | Float | No |
| `immunity_steps` | Steps fish is safe early in the episode | `200` | Integer | No |
| `distance_from_fish` | Initial distance from the fish (px) | `180.0` | Float | No |
| `epoch_num` | Number of predator attack epochs | `5` | Integer | No |
| `epoch_duration` | Duration of each epoch (steps) | `50` | Integer | No |
| `probability_per_epoch_step` | Probability of predator action per step during epochs | `0.07` | Float | No |
| `stim_file` | Path to predator stimulus location data | `./analysis/...` | String | Yes |

---

## 8. Salt source
Salt gradient configuration

| Name | Meaning | Example Value | Data Type | Optional |
| :--- | :--- | :--- | :--- | :--- |
| `enabled` | Whether salt gradients are enabled | `true` | Boolean | No |
| `concentration_decay` | concentration decay constant (1/pixels) | `0.002` | Float | No |

---

## 9. Rewards
| Name | Meaning | Example Value | Data Type | Optional |
| :--- | :--- | :--- | :--- | :--- |
| `predator_caught` | Penalty for being eaten | `-2` | Integer | No |
| `predator_avoidance` | Reward for avoiding predators | `2` | Integer | No |
| `consumption` | Reward for catching prey | `3` | Integer | No |
| `energy_use_factor` | Factor for translating energy use to negative rewards | `10` | Integer | No |
| `wall_touch` | Penalty for touching walls | `-2` | Integer | No |
| `salt_factor` | Factor for salt concentration influence | `-0.05` | Float | No |

---

## 10. Energy
| Name | Meaning | Example Value | Data Type | Optional |
| :--- | :--- | :--- | :--- | :--- |
| `distance_factor` | Energy cost per unit of movement | `0.00023` | Float | No |
| `angle_factor` | Energy cost per degree of turning | `0.0001` | Float | No |
| `baseline` | Passive energy loss per step | `0.00002` | Float | No |
