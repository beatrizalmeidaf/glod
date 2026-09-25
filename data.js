const GLOD_DATA = {
  "kappas": {
    "gsm8k|Mistral-7B-Instruct-v0.3": 0.2310022112120534,
    "gsm8k|OLMo-2-0425-1B-Instruct": 0.24294478270297923,
    "gsm8k|Phi-3.5-mini-instruct": 0.1957311922732073,
    "gsm8k|Qwen3-14B": 0.1819051579257201,
    "gsm8k|Qwen3-4B": 0.17151640048403718,
    "gsm8k|gemma-3-12b-it": 0.152800787908269,
    "gsm8k|gemma-3-1b-it": 0.14368205324406808,
    "gsm8k|gemma-3-4b-it": 0.12991289964028466,
    "mix|Mistral-7B-Instruct-v0.3": 0.27720843042681625,
    "mix|Mistral-Small-24B-Instruct-2501": 0.22471341602286043,
    "mix|OLMo-2-0425-1B": 0.19015426316785272,
    "mix|OLMo-2-0425-1B-Instruct": 0.35474816305476986,
    "mix|OLMo-2-1124-13B-Instruct": 0.29546286804149025,
    "mix|OLMo-2-1124-7B-Instruct": 0.28183682521685827,
    "mix|Phi-3.5-mini-instruct": 0.23012484375143322,
    "mix|Qwen2.5-7B-Instruct": 0.27966266617178404,
    "mix|Qwen3-14B": 0.22215392014521437,
    "mix|Qwen3-4B": 0.18926613964865396,
    "mix|Qwen3-8B": 0.22893364661133164,
    "mix|gemma-3-12b-it": 0.2109525894993956,
    "mix|gemma-3-1b-it": 0.19674534088249468,
    "mix|gemma-3-27b-it": 0.18827004433401512,
    "mix|gemma-3-4b-it": 0.2020328395755244,
    "mix|phi-4": 0.2650002885997288,
    "mmlu_en|Mistral-7B-Instruct-v0.3": 0.3180994250773795,
    "mmlu_en|OLMo-2-0425-1B-Instruct": 0.4523492987278337,
    "mmlu_en|Phi-3.5-mini-instruct": 0.3070802090021828,
    "mmlu_en|Qwen3-14B": 0.2641931078987132,
    "mmlu_en|Qwen3-4B": 0.21021450044264087,
    "mmlu_en|gemma-3-12b-it": 0.23843933155768562,
    "mmlu_en|gemma-3-1b-it": 0.25090689683955,
    "mmlu_en|gemma-3-4b-it": 0.22344256025687886,
    "wikitext|Mistral-7B-Instruct-v0.3": 0.37640192543765083,
    "wikitext|OLMo-2-0425-1B-Instruct": 0.564575557340622,
    "wikitext|OLMo-2-1124-7B-Instruct": 0.49414664548395265,
    "wikitext|Phi-3.5-mini-instruct": 0.4512207876098187,
    "wikitext|Qwen3-14B": 0.30980183709873044,
    "wikitext|Qwen3-4B": 0.27095034432122045,
    "wikitext|gemma-3-12b-it": 0.2984916967939766,
    "wikitext|gemma-3-1b-it": 0.3661336076437818,
    "wikitext|gemma-3-4b-it": 0.2891428258206391,
    "wikitext_nat|Mistral-7B-Instruct-v0.3": 0.4766522296732949,
    "wikitext_nat|OLMo-2-0425-1B": 0.5586909432203785,
    "wikitext_nat|OLMo-2-0425-1B-Instruct": 0.559590934791488,
    "wikitext_nat|OLMo-2-1124-7B": 0.48421539234999184,
    "wikitext_nat|OLMo-2-1124-7B-Instruct": 0.46811946948221944,
    "wikitext_nat|Qwen2.5-72B-Instruct": 0.42173316996245075,
    "wikitext_nat|Qwen3-4B": 0.39020450127294826,
    "wikitext_nat|gemma-3-4b-it": 0.40824448198316343
  },
  "measurements": {
    "gsm8k": {
      "Meta-Llama-3-8B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000439536,
          "flip": 0.00423842
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00233231,
          "flip": 0.0105406
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00572604,
          "flip": 0.0167318
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0106842,
          "flip": 0.023389
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.023993,
          "flip": 0.0346618
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0247259,
          "flip": 0.0345731
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0257838,
          "flip": 0.036082
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0361076,
          "flip": 0.0429167
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0412929,
          "flip": 0.0485976
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0428153,
          "flip": 0.0449361
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.18174,
          "flip": 0.096951
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.197053,
          "flip": 0.101145
        }
      ],
      "Mistral-7B-Instruct-v0.3": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000502648,
          "flip": 0.00477264
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00342743,
          "flip": 0.0128147
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00803637,
          "flip": 0.0204434
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.018254,
          "flip": 0.0312101
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0226336,
          "flip": 0.0364149
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0339961,
          "flip": 0.0421646
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0362923,
          "flip": 0.0443818
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.041816,
          "flip": 0.0475009
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0566346,
          "flip": 0.0515784
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0610925,
          "flip": 0.0588313
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.181348,
          "flip": 0.103082
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.187527,
          "flip": 0.106107
        }
      ],
      "OLMo-2-0425-1B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000335617,
          "flip": 0.00434627
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00259402,
          "flip": 0.0125113
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00600448,
          "flip": 0.0194949
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00939726,
          "flip": 0.0242631
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0315406,
          "flip": 0.0431462
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0315537,
          "flip": 0.0426609
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0350656,
          "flip": 0.0442222
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0416151,
          "flip": 0.0503196
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0451696,
          "flip": 0.0512691
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0465376,
          "flip": 0.0517754
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.19802,
          "flip": 0.114522
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.234309,
          "flip": 0.124818
        }
      ],
      "Phi-3.5-mini-instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00133529,
          "flip": 0.00538941
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00338461,
          "flip": 0.0110526
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00502331,
          "flip": 0.0145942
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00790682,
          "flip": 0.0176054
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0177418,
          "flip": 0.0251848
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0215563,
          "flip": 0.0292739
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0249179,
          "flip": 0.0294279
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.035196,
          "flip": 0.0371955
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0355605,
          "flip": 0.037435
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0472872,
          "flip": 0.0420716
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.15954,
          "flip": 0.0878901
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.232711,
          "flip": 0.106522
        }
      ],
      "Qwen3-14B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000567232,
          "flip": 0.00411396
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00220399,
          "flip": 0.00824715
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00399467,
          "flip": 0.0101696
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00723451,
          "flip": 0.0158022
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0148735,
          "flip": 0.0221846
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0155242,
          "flip": 0.0222231
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0203248,
          "flip": 0.0254143
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.029524,
          "flip": 0.0322773
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0311345,
          "flip": 0.0325849
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0317163,
          "flip": 0.0324503
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.142033,
          "flip": 0.0709947
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.173718,
          "flip": 0.0783575
        }
      ],
      "Qwen3-4B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000733537,
          "flip": 0.00432337
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00494094,
          "flip": 0.0120562
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00685651,
          "flip": 0.0133743
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0179159,
          "flip": 0.0228822
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0277982,
          "flip": 0.0303866
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0332205,
          "flip": 0.0306854
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0341813,
          "flip": 0.0323199
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0355107,
          "flip": 0.0346397
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0600098,
          "flip": 0.0454657
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0731696,
          "flip": 0.0504218
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.299123,
          "flip": 0.106714
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.318433,
          "flip": 0.117206
        }
      ],
      "gemma-3-12b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000398644,
          "flip": 0.00325842
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00175392,
          "flip": 0.00668393
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00439985,
          "flip": 0.0098588
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0071794,
          "flip": 0.0129292
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0178832,
          "flip": 0.0206158
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0193527,
          "flip": 0.0219525
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0248652,
          "flip": 0.024271
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0290839,
          "flip": 0.0252527
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0319523,
          "flip": 0.0272579
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0346122,
          "flip": 0.0284276
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.242035,
          "flip": 0.0902958
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.276701,
          "flip": 0.0910268
        }
      ],
      "gemma-3-1b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00226168,
          "flip": 0.00613258
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0219899,
          "flip": 0.0213066
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0259002,
          "flip": 0.0231408
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0747705,
          "flip": 0.0390188
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.126804,
          "flip": 0.0545448
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.157865,
          "flip": 0.0625857
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.190632,
          "flip": 0.0685886
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.198089,
          "flip": 0.0718865
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.262519,
          "flip": 0.0827621
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.27514,
          "flip": 0.0841146
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.652292,
          "flip": 0.174436
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.706683,
          "flip": 0.176863
        }
      ],
      "gemma-3-4b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000668845,
          "flip": 0.00360558
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00509168,
          "flip": 0.00971939
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0104277,
          "flip": 0.0132662
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0183549,
          "flip": 0.0174008
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0441288,
          "flip": 0.0283939
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0442278,
          "flip": 0.0271594
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0520413,
          "flip": 0.0310785
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.057677,
          "flip": 0.0351348
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0682395,
          "flip": 0.0364477
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0729428,
          "flip": 0.0379174
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.482934,
          "flip": 0.095783
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.679831,
          "flip": 0.137678
        }
      ]
    },
    "mix": {
      "Meta-Llama-3-8B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000588803,
          "flip": 0.00568269
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00368041,
          "flip": 0.0166447
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00826421,
          "flip": 0.0235731
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0153522,
          "flip": 0.0328268
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0392522,
          "flip": 0.0514408
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0405304,
          "flip": 0.05169
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.044163,
          "flip": 0.0555338
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0446722,
          "flip": 0.0583811
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0626985,
          "flip": 0.0651908
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0757592,
          "flip": 0.0724158
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.311424,
          "flip": 0.145876
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.318012,
          "flip": 0.145282
        }
      ],
      "Mistral-7B-Instruct-v0.3": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000536175,
          "flip": 0.006457
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00376447,
          "flip": 0.0170082
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00863889,
          "flip": 0.0257567
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0211462,
          "flip": 0.0406363
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0257038,
          "flip": 0.0461971
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0363782,
          "flip": 0.0521652
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0415191,
          "flip": 0.0579093
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0446756,
          "flip": 0.0581028
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0661188,
          "flip": 0.0697438
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0722345,
          "flip": 0.0715872
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0723746,
          "flip": 0.076435
        },
        {
          "family": "skip",
          "config": "skip28",
          "kl": 0.120755,
          "flip": 0.0878519
        },
        {
          "family": "skip",
          "config": "skip16",
          "kl": 0.130297,
          "flip": 0.0999613
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.165596,
          "flip": 0.112824
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.183454,
          "flip": 0.1203
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.196761,
          "flip": 0.121695
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.214604,
          "flip": 0.129924
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.223467,
          "flip": 0.131045
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.230023,
          "flip": 0.13299
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.338569,
          "flip": 0.16558
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.356853,
          "flip": 0.165326
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.564094,
          "flip": 0.215179
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 0.571716,
          "flip": 0.212399
        }
      ],
      "Mistral-Small-24B-Instruct-2501": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000279006,
          "flip": 0.0035349
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00310051,
          "flip": 0.012306
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00479448,
          "flip": 0.0152644
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0128511,
          "flip": 0.0254721
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.01536,
          "flip": 0.0280146
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0176653,
          "flip": 0.0291582
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0187333,
          "flip": 0.0318992
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0198853,
          "flip": 0.0322395
        },
        {
          "family": "skip",
          "config": "skip20",
          "kl": 0.0489579,
          "flip": 0.049725
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.101385,
          "flip": 0.0755846
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.14736,
          "flip": 0.086539
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.165183,
          "flip": 0.0803009
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.212064,
          "flip": 0.108788
        }
      ],
      "OLMo-2-0425-1B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00027494,
          "flip": 0.00342451
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00643271,
          "flip": 0.0152239
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00971671,
          "flip": 0.0206164
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0295828,
          "flip": 0.0346092
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0428829,
          "flip": 0.0393775
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0491594,
          "flip": 0.0400191
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0542325,
          "flip": 0.0470935
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0586528,
          "flip": 0.0470762
        },
        {
          "family": "skip",
          "config": "skip8",
          "kl": 0.152274,
          "flip": 0.0784689
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.266626,
          "flip": 0.0994408
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.308698,
          "flip": 0.117049
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.523787,
          "flip": 0.154094
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.558982,
          "flip": 0.164619
        }
      ],
      "OLMo-2-0425-1B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00070196,
          "flip": 0.00984562
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00567243,
          "flip": 0.0283477
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0130479,
          "flip": 0.0415894
        },
        {
          "family": "camada",
          "config": "lq3_1",
          "kl": 0.0185464,
          "flip": 0.0472568
        },
        {
          "family": "camada",
          "config": "lq3_0",
          "kl": 0.018587,
          "flip": 0.0461641
        },
        {
          "family": "camada",
          "config": "lq3_13",
          "kl": 0.0186424,
          "flip": 0.0483817
        },
        {
          "family": "camada",
          "config": "lq3_14",
          "kl": 0.0188989,
          "flip": 0.0493567
        },
        {
          "family": "camada",
          "config": "lq3_15",
          "kl": 0.0196664,
          "flip": 0.0525385
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0217804,
          "flip": 0.0546598
        },
        {
          "family": "camada",
          "config": "lq3_2",
          "kl": 0.0225596,
          "flip": 0.0530956
        },
        {
          "family": "camada",
          "config": "lq3_4",
          "kl": 0.0233237,
          "flip": 0.0530099
        },
        {
          "family": "camada",
          "config": "lq3_3",
          "kl": 0.024659,
          "flip": 0.0543598
        },
        {
          "family": "camada",
          "config": "lq3_12",
          "kl": 0.025614,
          "flip": 0.057006
        },
        {
          "family": "camada",
          "config": "lq3_5",
          "kl": 0.0271914,
          "flip": 0.0581417
        },
        {
          "family": "camada",
          "config": "lq3_11",
          "kl": 0.030526,
          "flip": 0.064409
        },
        {
          "family": "camada",
          "config": "lq3_6",
          "kl": 0.0327222,
          "flip": 0.062652
        },
        {
          "family": "camada",
          "config": "lq3_7",
          "kl": 0.0336922,
          "flip": 0.0657696
        },
        {
          "family": "camada",
          "config": "lq3_8",
          "kl": 0.0342829,
          "flip": 0.0656839
        },
        {
          "family": "camada",
          "config": "lq3_9",
          "kl": 0.0361566,
          "flip": 0.068148
        },
        {
          "family": "camada",
          "config": "lq3_10",
          "kl": 0.0381061,
          "flip": 0.0687158
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0623023,
          "flip": 0.0875928
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0692968,
          "flip": 0.0926388
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0878873,
          "flip": 0.103245
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0892856,
          "flip": 0.104884
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0917054,
          "flip": 0.107455
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0950192,
          "flip": 0.10768
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0994136,
          "flip": 0.110605
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.339155,
          "flip": 0.195123
        },
        {
          "family": "skip",
          "config": "skip8",
          "kl": 0.339527,
          "flip": 0.20048
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.386533,
          "flip": 0.211954
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.432894,
          "flip": 0.213743
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.448399,
          "flip": 0.21249
        },
        {
          "family": "skip",
          "config": "skip13",
          "kl": 0.449715,
          "flip": 0.209833
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.462543,
          "flip": 0.226524
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.465714,
          "flip": 0.227617
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.553276,
          "flip": 0.242305
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.597608,
          "flip": 0.250201
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.895071,
          "flip": 0.325473
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 1.15226,
          "flip": 0.364159
        }
      ],
      "OLMo-2-1124-13B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000397366,
          "flip": 0.00612455
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00270673,
          "flip": 0.0157743
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00648934,
          "flip": 0.0238186
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0106936,
          "flip": 0.0319394
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0324025,
          "flip": 0.0521648
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0366526,
          "flip": 0.0559789
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.036992,
          "flip": 0.0566414
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0443015,
          "flip": 0.0627915
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0448631,
          "flip": 0.0625366
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0454358,
          "flip": 0.0616532
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0494386,
          "flip": 0.0662997
        },
        {
          "family": "skip",
          "config": "skip36",
          "kl": 0.0813432,
          "flip": 0.0753718
        },
        {
          "family": "skip",
          "config": "skip20",
          "kl": 0.0935798,
          "flip": 0.0901693
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.175611,
          "flip": 0.119136
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.210409,
          "flip": 0.126968
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.220994,
          "flip": 0.133874
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.228503,
          "flip": 0.136898
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.233781,
          "flip": 0.138155
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.240377,
          "flip": 0.140567
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.253315,
          "flip": 0.144517
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.326108,
          "flip": 0.161965
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.611406,
          "flip": 0.220764
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 0.759117,
          "flip": 0.249764
        }
      ],
      "OLMo-2-1124-7B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000394963,
          "flip": 0.00613634
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00300267,
          "flip": 0.0165726
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00770424,
          "flip": 0.0256539
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.01149,
          "flip": 0.0300882
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0379154,
          "flip": 0.0553727
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0385651,
          "flip": 0.0552719
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0458559,
          "flip": 0.0600421
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0472484,
          "flip": 0.0603556
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0477261,
          "flip": 0.0616546
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0501669,
          "flip": 0.0636142
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0600003,
          "flip": 0.0648123
        },
        {
          "family": "skip",
          "config": "skip28",
          "kl": 0.0862273,
          "flip": 0.0743416
        },
        {
          "family": "skip",
          "config": "skip16",
          "kl": 0.151442,
          "flip": 0.1063
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.182753,
          "flip": 0.115706
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.237947,
          "flip": 0.131237
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.247094,
          "flip": 0.13173
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.253508,
          "flip": 0.138527
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.291138,
          "flip": 0.140923
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.292901,
          "flip": 0.143163
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.344393,
          "flip": 0.153129
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.36863,
          "flip": 0.154047
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.580206,
          "flip": 0.211245
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 0.731142,
          "flip": 0.232285
        }
      ],
      "Phi-3.5-mini-instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00174017,
          "flip": 0.00900248
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00759811,
          "flip": 0.0208245
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0118719,
          "flip": 0.025074
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0263583,
          "flip": 0.0376815
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0492685,
          "flip": 0.0508227
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0606514,
          "flip": 0.0576501
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.100108,
          "flip": 0.0603589
        },
        {
          "family": "skip",
          "config": "skip28",
          "kl": 0.10715,
          "flip": 0.0781524
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.113287,
          "flip": 0.0747185
        },
        {
          "family": "skip",
          "config": "skip16",
          "kl": 0.124225,
          "flip": 0.0780517
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.136796,
          "flip": 0.0792903
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.169102,
          "flip": 0.0817574
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.185369,
          "flip": 0.0887962
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.321219,
          "flip": 0.136568
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.339898,
          "flip": 0.135178
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.387295,
          "flip": 0.151219
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.403201,
          "flip": 0.141925
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.49214,
          "flip": 0.178851
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.647045,
          "flip": 0.179194
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.994209,
          "flip": 0.26947
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 1.2304,
          "flip": 0.312529
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 1.53487,
          "flip": 0.357159
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 1.85783,
          "flip": 0.380944
        }
      ],
      "Qwen2.5-7B-Instruct": [
        {
          "family": "oficial",
          "config": "ofc@Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
          "kl": 0.000596043,
          "flip": 0.00666169
        },
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000606689,
          "flip": 0.00663472
        },
        {
          "family": "gptq",
          "config": "gptq8",
          "kl": 0.000625407,
          "flip": 0.00675159
        },
        {
          "family": "oficial",
          "config": "ofc@Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
          "kl": 0.0283392,
          "flip": 0.0461464
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0320784,
          "flip": 0.0503088
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0379973,
          "flip": 0.054732
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0407268,
          "flip": 0.0562783
        },
        {
          "family": "oficial",
          "config": "ofc@Qwen/Qwen2.5-7B-Instruct-AWQ",
          "kl": 0.0425037,
          "flip": 0.0549657
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0431073,
          "flip": 0.0582291
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.192383,
          "flip": 0.120738
        }
      ],
      "Qwen3-14B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000756028,
          "flip": 0.00578292
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00406782,
          "flip": 0.0141843
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00651818,
          "flip": 0.0163348
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0153334,
          "flip": 0.0275664
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0273575,
          "flip": 0.034575
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0304024,
          "flip": 0.0378843
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0381455,
          "flip": 0.0433886
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0415606,
          "flip": 0.0454054
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0608203,
          "flip": 0.0544531
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0613309,
          "flip": 0.0542859
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.063813,
          "flip": 0.0551885
        },
        {
          "family": "skip",
          "config": "skip20",
          "kl": 0.0876835,
          "flip": 0.0590438
        },
        {
          "family": "skip",
          "config": "skip36",
          "kl": 0.0996306,
          "flip": 0.0596454
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.130268,
          "flip": 0.0822646
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.166008,
          "flip": 0.0914014
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.168529,
          "flip": 0.0958026
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.197133,
          "flip": 0.106098
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.243093,
          "flip": 0.10933
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.258898,
          "flip": 0.113274
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.301927,
          "flip": 0.129063
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.314881,
          "flip": 0.125497
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.494899,
          "flip": 0.171504
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 0.61385,
          "flip": 0.191717
        }
      ],
      "Qwen3-4B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00125561,
          "flip": 0.00683315
        },
        {
          "family": "camada",
          "config": "lq3_0",
          "kl": 0.00645469,
          "flip": 0.0139352
        },
        {
          "family": "camada",
          "config": "lq3_29",
          "kl": 0.00767179,
          "flip": 0.0169633
        },
        {
          "family": "camada",
          "config": "lq3_30",
          "kl": 0.00939687,
          "flip": 0.0198321
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00949197,
          "flip": 0.0197325
        },
        {
          "family": "camada",
          "config": "lq3_31",
          "kl": 0.0102514,
          "flip": 0.0194336
        },
        {
          "family": "camada",
          "config": "lq3_27",
          "kl": 0.0104028,
          "flip": 0.0204098
        },
        {
          "family": "camada",
          "config": "lq3_26",
          "kl": 0.0104718,
          "flip": 0.0209178
        },
        {
          "family": "camada",
          "config": "lq3_32",
          "kl": 0.010578,
          "flip": 0.0205293
        },
        {
          "family": "camada",
          "config": "lq3_33",
          "kl": 0.010882,
          "flip": 0.0200612
        },
        {
          "family": "camada",
          "config": "lq3_1",
          "kl": 0.0112258,
          "flip": 0.0181288
        },
        {
          "family": "camada",
          "config": "lq3_34",
          "kl": 0.0113795,
          "flip": 0.0216848
        },
        {
          "family": "camada",
          "config": "lq3_28",
          "kl": 0.0122035,
          "flip": 0.0216449
        },
        {
          "family": "camada",
          "config": "lq3_25",
          "kl": 0.0126245,
          "flip": 0.0231291
        },
        {
          "family": "camada",
          "config": "lq3_4",
          "kl": 0.0142922,
          "flip": 0.0213461
        },
        {
          "family": "camada",
          "config": "lq3_8",
          "kl": 0.0145355,
          "flip": 0.0216649
        },
        {
          "family": "camada",
          "config": "lq3_2",
          "kl": 0.0163188,
          "flip": 0.0225613
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.016402,
          "flip": 0.023926
        },
        {
          "family": "camada",
          "config": "lq3_3",
          "kl": 0.016627,
          "flip": 0.0217147
        },
        {
          "family": "camada",
          "config": "lq3_9",
          "kl": 0.016653,
          "flip": 0.0234578
        },
        {
          "family": "camada",
          "config": "lq3_7",
          "kl": 0.0178204,
          "flip": 0.0245435
        },
        {
          "family": "camada",
          "config": "lq3_11",
          "kl": 0.0189874,
          "flip": 0.0254699
        },
        {
          "family": "camada",
          "config": "lq3_10",
          "kl": 0.0190416,
          "flip": 0.0248623
        },
        {
          "family": "camada",
          "config": "lq3_24",
          "kl": 0.0191522,
          "flip": 0.0272529
        },
        {
          "family": "camada",
          "config": "lq3_12",
          "kl": 0.0215442,
          "flip": 0.0268744
        },
        {
          "family": "camada",
          "config": "lq3_13",
          "kl": 0.0217235,
          "flip": 0.0274621
        },
        {
          "family": "camada",
          "config": "lq3_5",
          "kl": 0.0217645,
          "flip": 0.0258185
        },
        {
          "family": "camada",
          "config": "lq3_35",
          "kl": 0.0221087,
          "flip": 0.0330501
        },
        {
          "family": "camada",
          "config": "lq3_14",
          "kl": 0.0225028,
          "flip": 0.0277609
        },
        {
          "family": "camada",
          "config": "lq3_23",
          "kl": 0.0235499,
          "flip": 0.030291
        },
        {
          "family": "camada",
          "config": "lq3_17",
          "kl": 0.0241205,
          "flip": 0.0293945
        },
        {
          "family": "camada",
          "config": "lq3_16",
          "kl": 0.0254868,
          "flip": 0.0290458
        },
        {
          "family": "camada",
          "config": "lq3_20",
          "kl": 0.0261709,
          "flip": 0.0307492
        },
        {
          "family": "camada",
          "config": "lq3_22",
          "kl": 0.0265985,
          "flip": 0.0318249
        },
        {
          "family": "camada",
          "config": "lq3_15",
          "kl": 0.0285075,
          "flip": 0.030779
        },
        {
          "family": "camada",
          "config": "lq3_18",
          "kl": 0.0315934,
          "flip": 0.0321437
        },
        {
          "family": "camada",
          "config": "lq3_21",
          "kl": 0.0319092,
          "flip": 0.0350024
        },
        {
          "family": "camada",
          "config": "lq3_19",
          "kl": 0.0339704,
          "flip": 0.0340163
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0375566,
          "flip": 0.0392159
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0712885,
          "flip": 0.050123
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0762697,
          "flip": 0.054715
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0790604,
          "flip": 0.0558007
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0853866,
          "flip": 0.0594065
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.133213,
          "flip": 0.0745072
        },
        {
          "family": "skip",
          "config": "skip18",
          "kl": 0.138788,
          "flip": 0.0740689
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.139325,
          "flip": 0.0755033
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.140581,
          "flip": 0.0771169
        },
        {
          "family": "camada",
          "config": "lq3_6",
          "kl": 0.154228,
          "flip": 0.0766488
        },
        {
          "family": "skip",
          "config": "skip32",
          "kl": 0.164031,
          "flip": 0.0780134
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.335983,
          "flip": 0.124919
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.39796,
          "flip": 0.130696
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.403176,
          "flip": 0.135816
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.45586,
          "flip": 0.150837
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.475664,
          "flip": 0.156644
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.520843,
          "flip": 0.167093
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.670057,
          "flip": 0.186258
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.689559,
          "flip": 0.197972
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 1.21165,
          "flip": 0.269501
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 1.66887,
          "flip": 0.294532
        }
      ],
      "Qwen3-8B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000895986,
          "flip": 0.00685677
        },
        {
          "family": "oficial",
          "config": "ofc@Qwen/Qwen3-8B-AWQ",
          "kl": 0.0369606,
          "flip": 0.0442208
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0397321,
          "flip": 0.0454175
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0557064,
          "flip": 0.0531866
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0891532,
          "flip": 0.0671356
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.101186,
          "flip": 0.0685187
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.509909,
          "flip": 0.170193
        }
      ],
      "gemma-3-12b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000906019,
          "flip": 0.00678876
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0056968,
          "flip": 0.0159221
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0121516,
          "flip": 0.0236557
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0221999,
          "flip": 0.0313776
        },
        {
          "family": "skip",
          "config": "skip44",
          "kl": 0.0332501,
          "flip": 0.0344454
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0494948,
          "flip": 0.0489677
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0566053,
          "flip": 0.0510206
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0699445,
          "flip": 0.0563163
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0790139,
          "flip": 0.0623352
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0837413,
          "flip": 0.0633617
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0848768,
          "flip": 0.0633034
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.087472,
          "flip": 0.0636417
        },
        {
          "family": "skip",
          "config": "skip22",
          "kl": 0.180434,
          "flip": 0.0946693
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.231539,
          "flip": 0.111979
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.258793,
          "flip": 0.118442
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.334853,
          "flip": 0.139636
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.341583,
          "flip": 0.140056
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.37212,
          "flip": 0.141596
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.443226,
          "flip": 0.16349
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.488836,
          "flip": 0.168704
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.621518,
          "flip": 0.175283
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 1.03955,
          "flip": 0.265695
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 1.1725,
          "flip": 0.248618
        }
      ],
      "gemma-3-1b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00278765,
          "flip": 0.0103878
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0286215,
          "flip": 0.035871
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0485387,
          "flip": 0.0430098
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.104061,
          "flip": 0.066968
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.219862,
          "flip": 0.10201
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.259102,
          "flip": 0.112409
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.278802,
          "flip": 0.11852
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.366424,
          "flip": 0.137019
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.457582,
          "flip": 0.141196
        },
        {
          "family": "skip",
          "config": "skip13",
          "kl": 0.599563,
          "flip": 0.185599
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.880131,
          "flip": 0.245759
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.913841,
          "flip": 0.248411
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 1.66312,
          "flip": 0.380126
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 2.1451,
          "flip": 0.440652
        }
      ],
      "gemma-3-27b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000728522,
          "flip": 0.00537108
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00812321,
          "flip": 0.0168327
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.01577,
          "flip": 0.0236427
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0365671,
          "flip": 0.0376471
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0503835,
          "flip": 0.0440106
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0587923,
          "flip": 0.0471613
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0614909,
          "flip": 0.0467023
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0668401,
          "flip": 0.0482901
        },
        {
          "family": "skip",
          "config": "skip30",
          "kl": 0.11666,
          "flip": 0.0625178
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.291697,
          "flip": 0.11577
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.845157,
          "flip": 0.210253
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.944961,
          "flip": 0.224283
        }
      ],
      "gemma-3-4b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00161424,
          "flip": 0.00768135
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0135448,
          "flip": 0.0240832
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0255434,
          "flip": 0.0322895
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0520501,
          "flip": 0.045831
        },
        {
          "family": "skip",
          "config": "skip30",
          "kl": 0.0805455,
          "flip": 0.0503519
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.108501,
          "flip": 0.0658646
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.122023,
          "flip": 0.0739531
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.125226,
          "flip": 0.0751851
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.161623,
          "flip": 0.0838092
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.175082,
          "flip": 0.0867339
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.180594,
          "flip": 0.0901407
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.181385,
          "flip": 0.088748
        },
        {
          "family": "skip",
          "config": "skip17",
          "kl": 0.308085,
          "flip": 0.117491
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.467367,
          "flip": 0.156262
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.499438,
          "flip": 0.172964
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.566633,
          "flip": 0.177646
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.720008,
          "flip": 0.201076
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.725756,
          "flip": 0.2097
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.808079,
          "flip": 0.206614
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.851284,
          "flip": 0.184749
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.946099,
          "flip": 0.223713
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 1.35929,
          "flip": 0.294366
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 1.43456,
          "flip": 0.331819
        }
      ],
      "phi-4": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000339602,
          "flip": 0.00421613
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00103427,
          "flip": 0.00830127
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00216265,
          "flip": 0.0120344
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00329858,
          "flip": 0.0154318
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.00865675,
          "flip": 0.0241097
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0122021,
          "flip": 0.0309783
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0129326,
          "flip": 0.0303889
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0130363,
          "flip": 0.030176
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0133442,
          "flip": 0.0306508
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0133518,
          "flip": 0.0307982
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0139289,
          "flip": 0.0317069
        },
        {
          "family": "skip",
          "config": "skip20",
          "kl": 0.0217517,
          "flip": 0.039034
        },
        {
          "family": "skip",
          "config": "skip36",
          "kl": 0.0361254,
          "flip": 0.0433893
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.0596903,
          "flip": 0.065133
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.0650764,
          "flip": 0.0652886
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.066036,
          "flip": 0.0680639
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.0680882,
          "flip": 0.0697094
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.073183,
          "flip": 0.073115
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.0869308,
          "flip": 0.0802702
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.121331,
          "flip": 0.0933934
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.137232,
          "flip": 0.100573
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.287878,
          "flip": 0.144585
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 0.429987,
          "flip": 0.176267
        }
      ]
    },
    "mmlu_en": {
      "Meta-Llama-3-8B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000863208,
          "flip": 0.00877193
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00560835,
          "flip": 0.0253376
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0133835,
          "flip": 0.0378644
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0206423,
          "flip": 0.0482141
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.053466,
          "flip": 0.0745614
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0545944,
          "flip": 0.0767702
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0587776,
          "flip": 0.0786318
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.061786,
          "flip": 0.0851319
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0893105,
          "flip": 0.0949767
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0965961,
          "flip": 0.100341
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.332507,
          "flip": 0.195033
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.369157,
          "flip": 0.19355
        }
      ],
      "Mistral-7B-Instruct-v0.3": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00084537,
          "flip": 0.00964736
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00576399,
          "flip": 0.0235853
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0141095,
          "flip": 0.0390718
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0278058,
          "flip": 0.0526289
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0419393,
          "flip": 0.0656528
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.057217,
          "flip": 0.075808
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0630613,
          "flip": 0.0780421
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0653502,
          "flip": 0.0796415
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0930491,
          "flip": 0.0925893
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.103933,
          "flip": 0.0995963
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.284731,
          "flip": 0.163421
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.288509,
          "flip": 0.167839
        }
      ],
      "OLMo-2-0425-1B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000800198,
          "flip": 0.0129798
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00671941,
          "flip": 0.0363392
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.015132,
          "flip": 0.0558196
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0269161,
          "flip": 0.074213
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0813392,
          "flip": 0.121593
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0830617,
          "flip": 0.127624
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0943369,
          "flip": 0.133699
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0999066,
          "flip": 0.133187
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.116005,
          "flip": 0.150664
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.122421,
          "flip": 0.149961
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.393452,
          "flip": 0.259021
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.44809,
          "flip": 0.270317
        }
      ],
      "Phi-3.5-mini-instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00311129,
          "flip": 0.0151995
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0107179,
          "flip": 0.0331366
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0150113,
          "flip": 0.0378133
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0318134,
          "flip": 0.0544957
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0638198,
          "flip": 0.0768529
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0704022,
          "flip": 0.0838965
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0944098,
          "flip": 0.0882596
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.137722,
          "flip": 0.112071
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.145548,
          "flip": 0.115065
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.151089,
          "flip": 0.110417
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.41386,
          "flip": 0.19996
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.512739,
          "flip": 0.232355
        }
      ],
      "Qwen3-14B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00130303,
          "flip": 0.00945752
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00746189,
          "flip": 0.0230111
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0128434,
          "flip": 0.0275733
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0284477,
          "flip": 0.0456226
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0546549,
          "flip": 0.060708
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0552391,
          "flip": 0.0608412
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0700392,
          "flip": 0.0728629
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0755593,
          "flip": 0.071431
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.114382,
          "flip": 0.0906124
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.119706,
          "flip": 0.0909454
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.339739,
          "flip": 0.154184
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.377236,
          "flip": 0.154717
        }
      ],
      "Qwen3-4B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00249014,
          "flip": 0.0105552
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0183897,
          "flip": 0.0285069
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0351973,
          "flip": 0.0376239
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0703489,
          "flip": 0.0587858
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.149102,
          "flip": 0.084519
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.15109,
          "flip": 0.0782526
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.164277,
          "flip": 0.0842622
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.165637,
          "flip": 0.0823874
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.279181,
          "flip": 0.115029
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.305727,
          "flip": 0.11223
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.617078,
          "flip": 0.193282
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.634456,
          "flip": 0.192871
        }
      ],
      "gemma-3-12b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.0020278,
          "flip": 0.0106441
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0124871,
          "flip": 0.0268083
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0285651,
          "flip": 0.0400515
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0482925,
          "flip": 0.0543839
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.113183,
          "flip": 0.0824051
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.121991,
          "flip": 0.086935
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.145924,
          "flip": 0.0968365
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.158056,
          "flip": 0.0977029
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.187999,
          "flip": 0.107654
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.201222,
          "flip": 0.108594
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.661216,
          "flip": 0.239492
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.835354,
          "flip": 0.25375
        }
      ],
      "gemma-3-1b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00484193,
          "flip": 0.0174591
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0530922,
          "flip": 0.0583872
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0853261,
          "flip": 0.0718524
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.180505,
          "flip": 0.105249
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.320208,
          "flip": 0.144922
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.4497,
          "flip": 0.178851
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.45709,
          "flip": 0.188741
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.531493,
          "flip": 0.18741
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.662155,
          "flip": 0.215938
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.686379,
          "flip": 0.212704
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 1.11732,
          "flip": 0.333054
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 1.2145,
          "flip": 0.34496
        }
      ],
      "gemma-3-4b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.003656,
          "flip": 0.0137976
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0320026,
          "flip": 0.0391226
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0573091,
          "flip": 0.0523895
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.115826,
          "flip": 0.0765353
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.236151,
          "flip": 0.109142
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.265337,
          "flip": 0.122203
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.281918,
          "flip": 0.130134
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.318638,
          "flip": 0.133288
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.363291,
          "flip": 0.145081
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.364982,
          "flip": 0.14458
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 1.37163,
          "flip": 0.299213
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 1.57569,
          "flip": 0.27377
        }
      ]
    },
    "wikitext": {
      "Meta-Llama-3-8B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.0011551,
          "flip": 0.0130997
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00771881,
          "flip": 0.0362512
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0160072,
          "flip": 0.0514902
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.029463,
          "flip": 0.0688685
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0681282,
          "flip": 0.103596
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0690802,
          "flip": 0.0980863
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0712622,
          "flip": 0.103508
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0743504,
          "flip": 0.108373
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.129655,
          "flip": 0.139906
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.130602,
          "flip": 0.140375
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.391724,
          "flip": 0.239018
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.481669,
          "flip": 0.261466
        }
      ],
      "Mistral-7B-Instruct-v0.3": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000989032,
          "flip": 0.0115934
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00738585,
          "flip": 0.0324244
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.01555,
          "flip": 0.0452606
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0302163,
          "flip": 0.0652755
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0476716,
          "flip": 0.0826006
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0674161,
          "flip": 0.0938972
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.068856,
          "flip": 0.0941013
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0730812,
          "flip": 0.0962716
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.114085,
          "flip": 0.124133
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.122786,
          "flip": 0.126767
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.311208,
          "flip": 0.202078
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.345065,
          "flip": 0.214246
        }
      ],
      "OLMo-2-0425-1B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000896695,
          "flip": 0.0167196
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00734597,
          "flip": 0.048389
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0155883,
          "flip": 0.07169
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0287807,
          "flip": 0.0935476
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0837956,
          "flip": 0.157746
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0857201,
          "flip": 0.15857
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0947672,
          "flip": 0.161062
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.113839,
          "flip": 0.176407
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.119145,
          "flip": 0.185549
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.12935,
          "flip": 0.190446
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.396923,
          "flip": 0.316041
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.434559,
          "flip": 0.327691
        }
      ],
      "OLMo-2-1124-7B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000638415,
          "flip": 0.0126714
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00479964,
          "flip": 0.0342342
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0122933,
          "flip": 0.0546808
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0184415,
          "flip": 0.0674305
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0605052,
          "flip": 0.116941
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0619103,
          "flip": 0.116745
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0621919,
          "flip": 0.116784
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0775553,
          "flip": 0.131688
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.081538,
          "flip": 0.133529
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.120067,
          "flip": 0.142636
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.361156,
          "flip": 0.264101
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.399084,
          "flip": 0.269271
        }
      ],
      "Phi-3.5-mini-instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00339301,
          "flip": 0.0228615
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0124314,
          "flip": 0.0502647
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0145113,
          "flip": 0.0544687
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0394463,
          "flip": 0.089697
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0566247,
          "flip": 0.104918
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0684315,
          "flip": 0.117898
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0733486,
          "flip": 0.120015
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.149521,
          "flip": 0.16267
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.161573,
          "flip": 0.173364
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.164724,
          "flip": 0.171646
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.466875,
          "flip": 0.295175
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.47362,
          "flip": 0.274553
        }
      ],
      "Qwen3-14B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00179271,
          "flip": 0.0123653
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0125827,
          "flip": 0.0354544
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0178737,
          "flip": 0.0414182
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0524334,
          "flip": 0.0706899
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0785077,
          "flip": 0.0861739
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0822142,
          "flip": 0.0857635
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0873392,
          "flip": 0.0896482
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0976988,
          "flip": 0.0991136
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.238582,
          "flip": 0.14655
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.241474,
          "flip": 0.150161
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.420025,
          "flip": 0.203152
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.537636,
          "flip": 0.221836
        }
      ],
      "Qwen3-4B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00343111,
          "flip": 0.0158711
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0321851,
          "flip": 0.0477351
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0460162,
          "flip": 0.0593414
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.107483,
          "flip": 0.0897432
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.177562,
          "flip": 0.118378
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.20148,
          "flip": 0.125476
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.225066,
          "flip": 0.134584
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.235066,
          "flip": 0.13696
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.398724,
          "flip": 0.185944
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.461101,
          "flip": 0.188016
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.775514,
          "flip": 0.263472
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.9733,
          "flip": 0.285131
        }
      ],
      "gemma-3-12b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00340199,
          "flip": 0.01741
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.021587,
          "flip": 0.0446216
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0440464,
          "flip": 0.0617574
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.086239,
          "flip": 0.0880552
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.173117,
          "flip": 0.126348
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.192128,
          "flip": 0.130072
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.193595,
          "flip": 0.133065
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.22201,
          "flip": 0.135853
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.288449,
          "flip": 0.161853
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.291954,
          "flip": 0.161214
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.933877,
          "flip": 0.32277
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 1.27533,
          "flip": 0.353615
        }
      ],
      "gemma-3-1b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00559823,
          "flip": 0.0273946
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0588265,
          "flip": 0.0844143
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0961825,
          "flip": 0.1028
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.189827,
          "flip": 0.151526
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.380441,
          "flip": 0.204821
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.394869,
          "flip": 0.22312
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.403591,
          "flip": 0.220001
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.52477,
          "flip": 0.237499
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.687083,
          "flip": 0.285813
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.751016,
          "flip": 0.288629
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 1.0143,
          "flip": 0.370791
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 1.34469,
          "flip": 0.421963
        }
      ],
      "gemma-3-4b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00415283,
          "flip": 0.0183363
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0338056,
          "flip": 0.0540094
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0622595,
          "flip": 0.0729757
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.112064,
          "flip": 0.100219
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.242025,
          "flip": 0.138934
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.257051,
          "flip": 0.14606
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.277397,
          "flip": 0.153686
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.35127,
          "flip": 0.165765
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.410272,
          "flip": 0.185166
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.411383,
          "flip": 0.187512
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 1.22004,
          "flip": 0.304982
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 1.24385,
          "flip": 0.312911
        }
      ]
    },
    "wikitext_nat": {
      "Meta-Llama-3-8B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00106339,
          "flip": 0.0139866
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00585272,
          "flip": 0.0362829
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0107515,
          "flip": 0.0479863
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0223623,
          "flip": 0.0668888
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.043874,
          "flip": 0.0914887
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0543478,
          "flip": 0.10418
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0554392,
          "flip": 0.105352
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.10168,
          "flip": 0.140154
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.102034,
          "flip": 0.143301
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.136695,
          "flip": 0.171953
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.302865,
          "flip": 0.233453
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.431263,
          "flip": 0.283414
        }
      ],
      "Mistral-7B-Instruct-v0.3": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000566489,
          "flip": 0.0116501
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00338515,
          "flip": 0.0277326
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00740014,
          "flip": 0.040257
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0138704,
          "flip": 0.0578643
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0226666,
          "flip": 0.071995
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0288686,
          "flip": 0.0808596
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0293697,
          "flip": 0.0843567
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0495953,
          "flip": 0.0983857
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0506703,
          "flip": 0.10971
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0561772,
          "flip": 0.11272
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.161333,
          "flip": 0.187968
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.191069,
          "flip": 0.20651
        }
      ],
      "OLMo-2-0425-1B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.0007368,
          "flip": 0.0150146
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00654375,
          "flip": 0.046875
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0132995,
          "flip": 0.0606893
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0273329,
          "flip": 0.0923665
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0707463,
          "flip": 0.138753
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0730034,
          "flip": 0.145711
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.113645,
          "flip": 0.182983
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.115169,
          "flip": 0.181213
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.120244,
          "flip": 0.184245
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.131274,
          "flip": 0.187703
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.337121,
          "flip": 0.281352
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.459569,
          "flip": 0.334839
        }
      ],
      "OLMo-2-0425-1B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000933117,
          "flip": 0.0177612
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00799253,
          "flip": 0.052653
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0187981,
          "flip": 0.0745646
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0306584,
          "flip": 0.0979818
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0839673,
          "flip": 0.152445
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.1013,
          "flip": 0.167867
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.108031,
          "flip": 0.172221
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.116856,
          "flip": 0.178446
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.133942,
          "flip": 0.19574
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.138633,
          "flip": 0.197449
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.412799,
          "flip": 0.316711
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.508667,
          "flip": 0.355652
        }
      ],
      "OLMo-2-1124-7B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.0003938,
          "flip": 0.0106812
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00318298,
          "flip": 0.0290324
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00876392,
          "flip": 0.0446981
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0124989,
          "flip": 0.0559489
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0434663,
          "flip": 0.100952
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0476021,
          "flip": 0.0960083
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0505171,
          "flip": 0.0983276
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0569853,
          "flip": 0.110311
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0621725,
          "flip": 0.115682
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.166988,
          "flip": 0.147563
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.271575,
          "flip": 0.240092
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.309904,
          "flip": 0.237345
        }
      ],
      "OLMo-2-1124-7B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000882019,
          "flip": 0.0149536
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0069585,
          "flip": 0.0408325
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0191488,
          "flip": 0.0647583
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0288626,
          "flip": 0.0795288
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0906345,
          "flip": 0.138977
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0987929,
          "flip": 0.133931
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.104025,
          "flip": 0.133809
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.12327,
          "flip": 0.158081
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.1264,
          "flip": 0.156352
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.204298,
          "flip": 0.169149
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.473337,
          "flip": 0.295227
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.546584,
          "flip": 0.299764
        }
      ],
      "Qwen2.5-72B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00163465,
          "flip": 0.0178833
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0091674,
          "flip": 0.0403849
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0128447,
          "flip": 0.0477905
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0321677,
          "flip": 0.0747884
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0517976,
          "flip": 0.0945841
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.116698,
          "flip": 0.142558
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.139042,
          "flip": 0.162781
        }
      ],
      "Qwen3-4B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00252814,
          "flip": 0.020284
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0196599,
          "flip": 0.0536702
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0315371,
          "flip": 0.0692952
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0714565,
          "flip": 0.0993652
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.101739,
          "flip": 0.118774
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.115092,
          "flip": 0.132507
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.145133,
          "flip": 0.145386
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.155536,
          "flip": 0.153564
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.266703,
          "flip": 0.197144
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.273788,
          "flip": 0.200317
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.430679,
          "flip": 0.236084
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.495304,
          "flip": 0.269104
        }
      ],
      "gemma-3-4b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.0028492,
          "flip": 0.0220133
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.021027,
          "flip": 0.0588582
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0412818,
          "flip": 0.0829468
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0750026,
          "flip": 0.109965
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.162519,
          "flip": 0.14919
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.164241,
          "flip": 0.161296
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.187155,
          "flip": 0.171794
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.195383,
          "flip": 0.166524
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.268732,
          "flip": 0.205831
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.280259,
          "flip": 0.207397
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.891845,
          "flip": 0.34906
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 1.06628,
          "flip": 0.354268
        }
      ]
    }
  }
};
