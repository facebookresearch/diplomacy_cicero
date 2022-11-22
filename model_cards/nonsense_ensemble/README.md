# Nonsense Ensemble Mappings

Since mapping between nonsense classifier numbers (and descriptions) in the paper and filenames here may be confusing, for convenience here we provide a table mapping between the two.

|Model Number	|Context	|Corruption Type	|Filename	|
|---	|---	|---	|---	|
|1	|messages, state	|location (entity)	|location	|
|2	|messages, state	|power (entity)	|power	|
|3	|dialogue history, state	|symbol (entity)	|symbol	|
|4	|messages, state	|weak	|hvwm	|
|5	|orders (1 M-phase), state, intents	|denoising (seed 1)	|hvm_no_msg_hist	|
|6	|messages, orders (2 M-phases), state, intents	|denoising (seed 1)	|hvm_2mphase_orderhistory	|
|7	|messages, orders (2 M-phases), state (2 M-phases), intents	|denoising (seeds 1, 2)	|humanvsmodel_nonsense_classifier_denoising_singleseed_seed2_2Mphasesorderhistory_2Mphasesstatehistory_150000steps	|
|8	|messages (no speaker), orders (2 M-phases), state (2 M-phases), intents	|denoising (seed 1)	|humanvsmodel_nonsense_classifier_denoising_singleseed_nospeaker_nobilat	|
|9	|messages, orders (4 M-phases), state, intents	|denoising (seed 3)	|humanvsmodel_nonsense_classifier_denoising_singleseed_seed3_4MPhasesOrderHistory	|
|10	|messages (no speaker, bilateral), orders (2 M-phases), state (2 M-phases), intents	|denoising (seed 1)	|humanvsmodel_nonsense_classifier_denoising_singleseed_nospeaker_bilateral	|
|11	|messages, orders (2 M-phases), state (2 M-phases), intents	|weak justifications	|humanvsmodel_nonsense_classifier_denoising_justifications	|
|12	|messages, orders, state, intents	|non-sequiturs	|nonsequitur_detector	|
|13	|messages, orders (2 M-phases), state, intents	|denoising (seeds 1,2,3)	|humanvsmodel_nonsense_classifier_denoising_singleseed_3seeds	|
|14	|messages, orders (2 M-phases), state, intents	|denoising (cardinals)	|humanvsmodel_nonsense_classifier_denoising_cardinals	|
|15	|messages, orders (2 M-phases), state (2 M-phases), intents	|denoising (seeds 1, 2)	|hvm_2seeds_2mphase_orderhistory_statehistory	|
|16	|messages, orders (2 M-phases), state, intents	|denoising (negations)	|humanvsmodel_nonsense_classifier_denoising_negations	|
|	|	|	|	|


