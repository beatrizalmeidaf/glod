const GLOD_DATA = {
  "kappas": {
    "gsm8k|Meta-Llama-3-8B-Instruct": 0.22377359294480884,
    "gsm8k|Mistral-7B-Instruct-v0.3": 0.2310022112120534,
    "gsm8k|OLMo-2-0425-1B-Instruct": 0.24294478270297923,
    "gsm8k|Phi-3.5-mini-instruct": 0.1957311922732073,
    "gsm8k|Qwen3-14B": 0.1819051579257201,
    "gsm8k|Qwen3-4B": 0.17151640048403718,
    "gsm8k|gemma-3-12b-it": 0.152800787908269,
    "gsm8k|gemma-3-1b-it": 0.14368205324406808,
    "gsm8k|gemma-3-4b-it": 0.12991289964028466,
    "mix|Meta-Llama-3-8B-Instruct": 0.2642579783324983,
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
    "mmlu_en|Meta-Llama-3-8B-Instruct": 0.33557919010238385,
    "mmlu_en|Mistral-7B-Instruct-v0.3": 0.3180994250773795,
    "mmlu_en|OLMo-2-0425-1B-Instruct": 0.4523492987278337,
    "mmlu_en|Phi-3.5-mini-instruct": 0.3070802090021828,
    "mmlu_en|Qwen3-14B": 0.2641931078987132,
    "mmlu_en|Qwen3-4B": 0.21021450044264087,
    "mmlu_en|gemma-3-12b-it": 0.23843933155768562,
    "mmlu_en|gemma-3-1b-it": 0.25090689683955,
    "mmlu_en|gemma-3-4b-it": 0.22344256025687886,
    "wikitext|Meta-Llama-3-8B-Instruct": 0.4040969072008553,
    "wikitext|Mistral-7B-Instruct-v0.3": 0.37640192543765083,
    "wikitext|OLMo-2-0425-1B-Instruct": 0.564575557340622,
    "wikitext|OLMo-2-1124-7B-Instruct": 0.49414664548395265,
    "wikitext|Phi-3.5-mini-instruct": 0.4512207876098187,
    "wikitext|Qwen3-14B": 0.30980183709873044,
    "wikitext|Qwen3-4B": 0.27095034432122045,
    "wikitext|gemma-3-12b-it": 0.2984916967939766,
    "wikitext|gemma-3-1b-it": 0.3661336076437818,
    "wikitext|gemma-3-4b-it": 0.2891428258206391,
    "wikitext_nat|Meta-Llama-3-8B-Instruct": 0.44729616016732704,
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
          "flip": 0.00423842,
          "tv": 0.0041672
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00233231,
          "flip": 0.0105406,
          "tv": 0.0101987
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00572604,
          "flip": 0.0167318,
          "tv": 0.0156385
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0106842,
          "flip": 0.023389,
          "tv": 0.0218188
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.023993,
          "flip": 0.0346618,
          "tv": 0.0336461
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0247259,
          "flip": 0.0345731,
          "tv": 0.0341672
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0257838,
          "flip": 0.036082,
          "tv": 0.0350034
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0361076,
          "flip": 0.0429167,
          "tv": 0.0418105
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0412929,
          "flip": 0.0485976,
          "tv": 0.0451371
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0428153,
          "flip": 0.0449361,
          "tv": 0.044932
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.18174,
          "flip": 0.096951,
          "tv": 0.103262
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.197053,
          "flip": 0.101145,
          "tv": 0.105246
        }
      ],
      "Mistral-7B-Instruct-v0.3": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000502648,
          "flip": 0.00477264,
          "tv": 0.00503353
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00342743,
          "flip": 0.0128147,
          "tv": 0.0134275
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00803637,
          "flip": 0.0204434,
          "tv": 0.0201727
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.018254,
          "flip": 0.0312101,
          "tv": 0.0308195
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0226336,
          "flip": 0.0364149,
          "tv": 0.0363476
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0339961,
          "flip": 0.0421646,
          "tv": 0.0418459
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0362923,
          "flip": 0.0443818,
          "tv": 0.0447974
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.041816,
          "flip": 0.0475009,
          "tv": 0.0467584
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0566346,
          "flip": 0.0515784,
          "tv": 0.0529211
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0610925,
          "flip": 0.0588313,
          "tv": 0.0585256
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.181348,
          "flip": 0.103082,
          "tv": 0.103511
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.187527,
          "flip": 0.106107,
          "tv": 0.10603
        }
      ],
      "OLMo-2-0425-1B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000335617,
          "flip": 0.00434627,
          "tv": 0.00420392
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00259402,
          "flip": 0.0125113,
          "tv": 0.0116956
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00600448,
          "flip": 0.0194949,
          "tv": 0.0178993
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00939726,
          "flip": 0.0242631,
          "tv": 0.0224129
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0315406,
          "flip": 0.0431462,
          "tv": 0.0406458
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0315537,
          "flip": 0.0426609,
          "tv": 0.0400977
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0350656,
          "flip": 0.0442222,
          "tv": 0.0441142
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0416151,
          "flip": 0.0503196,
          "tv": 0.0481084
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0451696,
          "flip": 0.0512691,
          "tv": 0.0479244
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0465376,
          "flip": 0.0517754,
          "tv": 0.0491251
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.19802,
          "flip": 0.114522,
          "tv": 0.114646
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.234309,
          "flip": 0.124818,
          "tv": 0.123347
        }
      ],
      "Phi-3.5-mini-instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00133529,
          "flip": 0.00538941,
          "tv": 0.00628054
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00338461,
          "flip": 0.0110526,
          "tv": 0.0113814
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00502331,
          "flip": 0.0145942,
          "tv": 0.0139216
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00790682,
          "flip": 0.0176054,
          "tv": 0.0180191
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0177418,
          "flip": 0.0251848,
          "tv": 0.0270062
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0215563,
          "flip": 0.0292739,
          "tv": 0.0296699
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0249179,
          "flip": 0.0294279,
          "tv": 0.030155
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.035196,
          "flip": 0.0371955,
          "tv": 0.0390212
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0355605,
          "flip": 0.037435,
          "tv": 0.0381765
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0472872,
          "flip": 0.0420716,
          "tv": 0.0424591
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.15954,
          "flip": 0.0878901,
          "tv": 0.0931553
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.232711,
          "flip": 0.106522,
          "tv": 0.116855
        }
      ],
      "Qwen3-14B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000567232,
          "flip": 0.00411396,
          "tv": 0.00396723
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00220399,
          "flip": 0.00824715,
          "tv": 0.00798168
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00399467,
          "flip": 0.0101696,
          "tv": 0.0105419
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00723451,
          "flip": 0.0158022,
          "tv": 0.0147861
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0148735,
          "flip": 0.0221846,
          "tv": 0.0219788
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0155242,
          "flip": 0.0222231,
          "tv": 0.0215071
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0203248,
          "flip": 0.0254143,
          "tv": 0.0248685
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.029524,
          "flip": 0.0322773,
          "tv": 0.0312456
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0311345,
          "flip": 0.0325849,
          "tv": 0.0320341
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0317163,
          "flip": 0.0324503,
          "tv": 0.0331387
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.142033,
          "flip": 0.0709947,
          "tv": 0.069395
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.173718,
          "flip": 0.0783575,
          "tv": 0.0775572
        }
      ],
      "Qwen3-4B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000733537,
          "flip": 0.00432337,
          "tv": 0.00450181
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00494094,
          "flip": 0.0120562,
          "tv": 0.0120376
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00685651,
          "flip": 0.0133743,
          "tv": 0.0139598
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0179159,
          "flip": 0.0228822,
          "tv": 0.0232952
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0277982,
          "flip": 0.0303866,
          "tv": 0.0309555
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0332205,
          "flip": 0.0306854,
          "tv": 0.0314646
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0341813,
          "flip": 0.0323199,
          "tv": 0.0321261
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0355107,
          "flip": 0.0346397,
          "tv": 0.0342732
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0600098,
          "flip": 0.0454657,
          "tv": 0.0458386
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0731696,
          "flip": 0.0504218,
          "tv": 0.0501232
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.299123,
          "flip": 0.106714,
          "tv": 0.106548
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.318433,
          "flip": 0.117206,
          "tv": 0.11884
        }
      ],
      "gemma-3-12b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000398644,
          "flip": 0.00325842,
          "tv": 0.002981
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00175392,
          "flip": 0.00668393,
          "tv": 0.00629087
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00439985,
          "flip": 0.0098588,
          "tv": 0.00973344
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0071794,
          "flip": 0.0129292,
          "tv": 0.0126741
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0178832,
          "flip": 0.0206158,
          "tv": 0.0209811
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0193527,
          "flip": 0.0219525,
          "tv": 0.0218675
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0248652,
          "flip": 0.024271,
          "tv": 0.0239643
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0290839,
          "flip": 0.0252527,
          "tv": 0.0254427
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0319523,
          "flip": 0.0272579,
          "tv": 0.0261662
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0346122,
          "flip": 0.0284276,
          "tv": 0.0278796
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.242035,
          "flip": 0.0902958,
          "tv": 0.101542
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.276701,
          "flip": 0.0910268,
          "tv": 0.0950403
        }
      ],
      "gemma-3-1b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00226168,
          "flip": 0.00613258,
          "tv": 0.00632955
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0219899,
          "flip": 0.0213066,
          "tv": 0.0206728
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0259002,
          "flip": 0.0231408,
          "tv": 0.0232151
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0747705,
          "flip": 0.0390188,
          "tv": 0.0389327
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.126804,
          "flip": 0.0545448,
          "tv": 0.0547834
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.157865,
          "flip": 0.0625857,
          "tv": 0.0623774
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.190632,
          "flip": 0.0685886,
          "tv": 0.0683498
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.198089,
          "flip": 0.0718865,
          "tv": 0.0736961
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.262519,
          "flip": 0.0827621,
          "tv": 0.0845716
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.27514,
          "flip": 0.0841146,
          "tv": 0.0853986
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.652292,
          "flip": 0.174436,
          "tv": 0.190361
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.706683,
          "flip": 0.176863,
          "tv": 0.192827
        }
      ],
      "gemma-3-4b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000668845,
          "flip": 0.00360558,
          "tv": 0.00347121
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00509168,
          "flip": 0.00971939,
          "tv": 0.00915039
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0104277,
          "flip": 0.0132662,
          "tv": 0.0130856
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0183549,
          "flip": 0.0174008,
          "tv": 0.0176851
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0441288,
          "flip": 0.0283939,
          "tv": 0.029069
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0442278,
          "flip": 0.0271594,
          "tv": 0.027519
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0520413,
          "flip": 0.0310785,
          "tv": 0.0310191
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.057677,
          "flip": 0.0351348,
          "tv": 0.0340789
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0682395,
          "flip": 0.0364477,
          "tv": 0.0370248
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0729428,
          "flip": 0.0379174,
          "tv": 0.0382311
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.482934,
          "flip": 0.095783,
          "tv": 0.0973337
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.679831,
          "flip": 0.137678,
          "tv": 0.141555
        }
      ]
    },
    "mix": {
      "Meta-Llama-3-8B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000588803,
          "flip": 0.00568269,
          "tv": 0.00584349
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00368041,
          "flip": 0.0166447,
          "tv": 0.0150825
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00826421,
          "flip": 0.0235731,
          "tv": 0.0219818
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0153522,
          "flip": 0.0328268,
          "tv": 0.0305611
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0392522,
          "flip": 0.0514408,
          "tv": 0.0494996
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0405304,
          "flip": 0.05169,
          "tv": 0.0491419
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.044163,
          "flip": 0.0555338,
          "tv": 0.0529794
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0446722,
          "flip": 0.0583811,
          "tv": 0.0534911
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0626985,
          "flip": 0.0651908,
          "tv": 0.0626296
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0757592,
          "flip": 0.0724158,
          "tv": 0.0688594
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.311424,
          "flip": 0.145876,
          "tv": 0.14926
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.318012,
          "flip": 0.145282,
          "tv": 0.146271
        }
      ],
      "Mistral-7B-Instruct-v0.3": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000536175,
          "flip": 0.006457,
          "tv": 0.00590508
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00376447,
          "flip": 0.0170082,
          "tv": 0.0157693
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00863889,
          "flip": 0.0257567,
          "tv": 0.0235147
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0211462,
          "flip": 0.0406363,
          "tv": 0.037199
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0257038,
          "flip": 0.0461971,
          "tv": 0.0430363
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0363782,
          "flip": 0.0521652,
          "tv": 0.048648
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0415191,
          "flip": 0.0579093,
          "tv": 0.0544472
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0446756,
          "flip": 0.0581028,
          "tv": 0.0540379
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0661188,
          "flip": 0.0697438,
          "tv": 0.0652588
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0722345,
          "flip": 0.0715872,
          "tv": 0.0685025
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0723746,
          "flip": 0.076435,
          "tv": 0.0699922
        },
        {
          "family": "skip",
          "config": "skip28",
          "kl": 0.120755,
          "flip": 0.0878519,
          "tv": 0.0864003
        },
        {
          "family": "skip",
          "config": "skip16",
          "kl": 0.130297,
          "flip": 0.0999613,
          "tv": 0.0945118
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.165596,
          "flip": 0.112824,
          "tv": 0.108277
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.183454,
          "flip": 0.1203,
          "tv": 0.119067
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.196761,
          "flip": 0.121695,
          "tv": 0.124664
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.214604,
          "flip": 0.129924,
          "tv": 0.130001
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.223467,
          "flip": 0.131045,
          "tv": 0.126135
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.230023,
          "flip": 0.13299,
          "tv": 0.128689
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.338569,
          "flip": 0.16558,
          "tv": 0.172372
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.356853,
          "flip": 0.165326,
          "tv": 0.168748
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.564094,
          "flip": 0.215179,
          "tv": 0.226981
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 0.571716,
          "flip": 0.212399,
          "tv": 0.221641
        }
      ],
      "Mistral-Small-24B-Instruct-2501": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000279006,
          "flip": 0.0035349,
          "tv": 0.00341278
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00310051,
          "flip": 0.012306,
          "tv": 0.012038
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00479448,
          "flip": 0.0152644,
          "tv": 0.0153071
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0128511,
          "flip": 0.0254721,
          "tv": 0.0254521
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.01536,
          "flip": 0.0280146,
          "tv": 0.0271014
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0176653,
          "flip": 0.0291582,
          "tv": 0.0311131
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0187333,
          "flip": 0.0318992,
          "tv": 0.0304497
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0198853,
          "flip": 0.0322395,
          "tv": 0.0320487
        },
        {
          "family": "skip",
          "config": "skip20",
          "kl": 0.0489579,
          "flip": 0.049725,
          "tv": 0.0485472
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.101385,
          "flip": 0.0755846,
          "tv": 0.0768578
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.14736,
          "flip": 0.086539,
          "tv": 0.0863788
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.165183,
          "flip": 0.0803009,
          "tv": 0.0814863
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.212064,
          "flip": 0.108788,
          "tv": 0.144445
        }
      ],
      "OLMo-2-0425-1B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00027494,
          "flip": 0.00342451,
          "tv": 0.0038513
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00643271,
          "flip": 0.0152239,
          "tv": 0.0182622
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00971671,
          "flip": 0.0206164,
          "tv": 0.0232524
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0295828,
          "flip": 0.0346092,
          "tv": 0.0427204
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0428829,
          "flip": 0.0393775,
          "tv": 0.0511943
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0491594,
          "flip": 0.0400191,
          "tv": 0.0548965
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0542325,
          "flip": 0.0470935,
          "tv": 0.0567573
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0586528,
          "flip": 0.0470762,
          "tv": 0.0636721
        },
        {
          "family": "skip",
          "config": "skip8",
          "kl": 0.152274,
          "flip": 0.0784689,
          "tv": 0.103309
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.266626,
          "flip": 0.0994408,
          "tv": 0.161338
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.308698,
          "flip": 0.117049,
          "tv": 0.185863
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.523787,
          "flip": 0.154094,
          "tv": 0.248861
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.558982,
          "flip": 0.164619,
          "tv": 0.277569
        }
      ],
      "OLMo-2-0425-1B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00070196,
          "flip": 0.00984562,
          "tv": 0.00805245
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00567243,
          "flip": 0.0283477,
          "tv": 0.0225961
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0130479,
          "flip": 0.0415894,
          "tv": 0.0338026
        },
        {
          "family": "camada",
          "config": "lq3_1",
          "kl": 0.0185464,
          "flip": 0.0472568,
          "tv": 0.0378336
        },
        {
          "family": "camada",
          "config": "lq3_0",
          "kl": 0.018587,
          "flip": 0.0461641,
          "tv": 0.0375221
        },
        {
          "family": "camada",
          "config": "lq3_13",
          "kl": 0.0186424,
          "flip": 0.0483817,
          "tv": 0.0409424
        },
        {
          "family": "camada",
          "config": "lq3_14",
          "kl": 0.0188989,
          "flip": 0.0493567,
          "tv": 0.0412845
        },
        {
          "family": "camada",
          "config": "lq3_15",
          "kl": 0.0196664,
          "flip": 0.0525385,
          "tv": 0.0442107
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0217804,
          "flip": 0.0546598,
          "tv": 0.0444933
        },
        {
          "family": "camada",
          "config": "lq3_2",
          "kl": 0.0225596,
          "flip": 0.0530956,
          "tv": 0.0427147
        },
        {
          "family": "camada",
          "config": "lq3_4",
          "kl": 0.0233237,
          "flip": 0.0530099,
          "tv": 0.0436361
        },
        {
          "family": "camada",
          "config": "lq3_3",
          "kl": 0.024659,
          "flip": 0.0543598,
          "tv": 0.0442994
        },
        {
          "family": "camada",
          "config": "lq3_12",
          "kl": 0.025614,
          "flip": 0.057006,
          "tv": 0.0485604
        },
        {
          "family": "camada",
          "config": "lq3_5",
          "kl": 0.0271914,
          "flip": 0.0581417,
          "tv": 0.0478409
        },
        {
          "family": "camada",
          "config": "lq3_11",
          "kl": 0.030526,
          "flip": 0.064409,
          "tv": 0.0535161
        },
        {
          "family": "camada",
          "config": "lq3_6",
          "kl": 0.0327222,
          "flip": 0.062652,
          "tv": 0.0522142
        },
        {
          "family": "camada",
          "config": "lq3_7",
          "kl": 0.0336922,
          "flip": 0.0657696,
          "tv": 0.0538496
        },
        {
          "family": "camada",
          "config": "lq3_8",
          "kl": 0.0342829,
          "flip": 0.0656839,
          "tv": 0.0549934
        },
        {
          "family": "camada",
          "config": "lq3_9",
          "kl": 0.0361566,
          "flip": 0.068148,
          "tv": 0.0566875
        },
        {
          "family": "camada",
          "config": "lq3_10",
          "kl": 0.0381061,
          "flip": 0.0687158,
          "tv": 0.0587032
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0623023,
          "flip": 0.0875928,
          "tv": 0.0737222
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0692968,
          "flip": 0.0926388,
          "tv": 0.0791738
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0878873,
          "flip": 0.103245,
          "tv": 0.0894254
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0892856,
          "flip": 0.104884,
          "tv": 0.0902039
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0917054,
          "flip": 0.107455,
          "tv": 0.0925262
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0950192,
          "flip": 0.10768,
          "tv": 0.0936026
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0994136,
          "flip": 0.110605,
          "tv": 0.0944552
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.339155,
          "flip": 0.195123,
          "tv": 0.182933
        },
        {
          "family": "skip",
          "config": "skip8",
          "kl": 0.339527,
          "flip": 0.20048,
          "tv": 0.183229
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.386533,
          "flip": 0.211954,
          "tv": 0.202062
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.432894,
          "flip": 0.213743,
          "tv": 0.201414
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.448399,
          "flip": 0.21249,
          "tv": 0.19846
        },
        {
          "family": "skip",
          "config": "skip13",
          "kl": 0.449715,
          "flip": 0.209833,
          "tv": 0.204351
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.462543,
          "flip": 0.226524,
          "tv": 0.215114
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.465714,
          "flip": 0.227617,
          "tv": 0.216693
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.553276,
          "flip": 0.242305,
          "tv": 0.234696
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.597608,
          "flip": 0.250201,
          "tv": 0.239598
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.895071,
          "flip": 0.325473,
          "tv": 0.330762
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 1.15226,
          "flip": 0.364159,
          "tv": 0.368138
        }
      ],
      "OLMo-2-1124-13B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000397366,
          "flip": 0.00612455,
          "tv": 0.00536141
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00270673,
          "flip": 0.0157743,
          "tv": 0.0138943
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00648934,
          "flip": 0.0238186,
          "tv": 0.0211125
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0106936,
          "flip": 0.0319394,
          "tv": 0.0276004
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0324025,
          "flip": 0.0521648,
          "tv": 0.0472715
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0366526,
          "flip": 0.0559789,
          "tv": 0.0507691
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.036992,
          "flip": 0.0566414,
          "tv": 0.0507266
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0443015,
          "flip": 0.0627915,
          "tv": 0.0565019
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0448631,
          "flip": 0.0625366,
          "tv": 0.0562944
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0454358,
          "flip": 0.0616532,
          "tv": 0.0564292
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0494386,
          "flip": 0.0662997,
          "tv": 0.0592947
        },
        {
          "family": "skip",
          "config": "skip36",
          "kl": 0.0813432,
          "flip": 0.0753718,
          "tv": 0.069909
        },
        {
          "family": "skip",
          "config": "skip20",
          "kl": 0.0935798,
          "flip": 0.0901693,
          "tv": 0.0818145
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.175611,
          "flip": 0.119136,
          "tv": 0.111814
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.210409,
          "flip": 0.126968,
          "tv": 0.119885
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.220994,
          "flip": 0.133874,
          "tv": 0.125373
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.228503,
          "flip": 0.136898,
          "tv": 0.129344
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.233781,
          "flip": 0.138155,
          "tv": 0.129171
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.240377,
          "flip": 0.140567,
          "tv": 0.132125
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.253315,
          "flip": 0.144517,
          "tv": 0.135534
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.326108,
          "flip": 0.161965,
          "tv": 0.153551
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.611406,
          "flip": 0.220764,
          "tv": 0.215397
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 0.759117,
          "flip": 0.249764,
          "tv": 0.246972
        }
      ],
      "OLMo-2-1124-7B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000394963,
          "flip": 0.00613634,
          "tv": 0.00520195
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00300267,
          "flip": 0.0165726,
          "tv": 0.0140983
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00770424,
          "flip": 0.0256539,
          "tv": 0.0223874
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.01149,
          "flip": 0.0300882,
          "tv": 0.0273939
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0379154,
          "flip": 0.0553727,
          "tv": 0.0497104
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0385651,
          "flip": 0.0552719,
          "tv": 0.0501966
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0458559,
          "flip": 0.0600421,
          "tv": 0.0548053
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0472484,
          "flip": 0.0603556,
          "tv": 0.0558242
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0477261,
          "flip": 0.0616546,
          "tv": 0.0553514
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0501669,
          "flip": 0.0636142,
          "tv": 0.0575318
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0600003,
          "flip": 0.0648123,
          "tv": 0.060422
        },
        {
          "family": "skip",
          "config": "skip28",
          "kl": 0.0862273,
          "flip": 0.0743416,
          "tv": 0.069764
        },
        {
          "family": "skip",
          "config": "skip16",
          "kl": 0.151442,
          "flip": 0.1063,
          "tv": 0.0997286
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.182753,
          "flip": 0.115706,
          "tv": 0.111194
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.237947,
          "flip": 0.131237,
          "tv": 0.12621
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.247094,
          "flip": 0.13173,
          "tv": 0.127595
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.253508,
          "flip": 0.138527,
          "tv": 0.132282
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.291138,
          "flip": 0.140923,
          "tv": 0.137174
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.292901,
          "flip": 0.143163,
          "tv": 0.13805
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.344393,
          "flip": 0.153129,
          "tv": 0.147359
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.36863,
          "flip": 0.154047,
          "tv": 0.152845
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.580206,
          "flip": 0.211245,
          "tv": 0.210845
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 0.731142,
          "flip": 0.232285,
          "tv": 0.233178
        }
      ],
      "Phi-3.5-mini-instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00174017,
          "flip": 0.00900248,
          "tv": 0.00891773
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00759811,
          "flip": 0.0208245,
          "tv": 0.0193423
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0118719,
          "flip": 0.025074,
          "tv": 0.0234484
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0263583,
          "flip": 0.0376815,
          "tv": 0.0349842
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0492685,
          "flip": 0.0508227,
          "tv": 0.0478328
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0606514,
          "flip": 0.0576501,
          "tv": 0.0553039
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.100108,
          "flip": 0.0603589,
          "tv": 0.0581383
        },
        {
          "family": "skip",
          "config": "skip28",
          "kl": 0.10715,
          "flip": 0.0781524,
          "tv": 0.0829933
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.113287,
          "flip": 0.0747185,
          "tv": 0.0717558
        },
        {
          "family": "skip",
          "config": "skip16",
          "kl": 0.124225,
          "flip": 0.0780517,
          "tv": 0.0761871
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.136796,
          "flip": 0.0792903,
          "tv": 0.0778696
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.169102,
          "flip": 0.0817574,
          "tv": 0.0792235
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.185369,
          "flip": 0.0887962,
          "tv": 0.0863543
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.321219,
          "flip": 0.136568,
          "tv": 0.142156
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.339898,
          "flip": 0.135178,
          "tv": 0.141084
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.387295,
          "flip": 0.151219,
          "tv": 0.156611
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.403201,
          "flip": 0.141925,
          "tv": 0.144878
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.49214,
          "flip": 0.178851,
          "tv": 0.195155
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.647045,
          "flip": 0.179194,
          "tv": 0.182719
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.994209,
          "flip": 0.26947,
          "tv": 0.302889
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 1.2304,
          "flip": 0.312529,
          "tv": 0.352794
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 1.53487,
          "flip": 0.357159,
          "tv": 0.425215
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 1.85783,
          "flip": 0.380944,
          "tv": 0.442321
        }
      ],
      "Qwen2.5-7B-Instruct": [
        {
          "family": "oficial",
          "config": "ofc@Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
          "kl": 0.000596043,
          "flip": 0.00666169,
          "tv": 0.00607429
        },
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000606689,
          "flip": 0.00663472,
          "tv": 0.00616368
        },
        {
          "family": "gptq",
          "config": "gptq8",
          "kl": 0.000625407,
          "flip": 0.00675159,
          "tv": 0.00628432
        },
        {
          "family": "oficial",
          "config": "ofc@Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
          "kl": 0.0283392,
          "flip": 0.0461464,
          "tv": 0.0421112
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0320784,
          "flip": 0.0503088,
          "tv": 0.0464548
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0379973,
          "flip": 0.054732,
          "tv": 0.0502452
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0407268,
          "flip": 0.0562783,
          "tv": 0.0523172
        },
        {
          "family": "oficial",
          "config": "ofc@Qwen/Qwen2.5-7B-Instruct-AWQ",
          "kl": 0.0425037,
          "flip": 0.0549657,
          "tv": 0.0511669
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0431073,
          "flip": 0.0582291,
          "tv": 0.0564679
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.192383,
          "flip": 0.120738,
          "tv": 0.119797
        }
      ],
      "Qwen3-14B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000756028,
          "flip": 0.00578292,
          "tv": 0.00550228
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00406782,
          "flip": 0.0141843,
          "tv": 0.0129767
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00651818,
          "flip": 0.0163348,
          "tv": 0.0160458
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0153334,
          "flip": 0.0275664,
          "tv": 0.0254002
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0273575,
          "flip": 0.034575,
          "tv": 0.033195
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0304024,
          "flip": 0.0378843,
          "tv": 0.035753
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0381455,
          "flip": 0.0433886,
          "tv": 0.0427045
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0415606,
          "flip": 0.0454054,
          "tv": 0.0442191
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0608203,
          "flip": 0.0544531,
          "tv": 0.0528955
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0613309,
          "flip": 0.0542859,
          "tv": 0.0517755
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.063813,
          "flip": 0.0551885,
          "tv": 0.0531213
        },
        {
          "family": "skip",
          "config": "skip20",
          "kl": 0.0876835,
          "flip": 0.0590438,
          "tv": 0.0569112
        },
        {
          "family": "skip",
          "config": "skip36",
          "kl": 0.0996306,
          "flip": 0.0596454,
          "tv": 0.0578382
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.130268,
          "flip": 0.0822646,
          "tv": 0.0834253
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.166008,
          "flip": 0.0914014,
          "tv": 0.0936531
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.168529,
          "flip": 0.0958026,
          "tv": 0.098845
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.197133,
          "flip": 0.106098,
          "tv": 0.104222
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.243093,
          "flip": 0.10933,
          "tv": 0.105496
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.258898,
          "flip": 0.113274,
          "tv": 0.111708
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.301927,
          "flip": 0.129063,
          "tv": 0.130547
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.314881,
          "flip": 0.125497,
          "tv": 0.125009
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.494899,
          "flip": 0.171504,
          "tv": 0.177348
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 0.61385,
          "flip": 0.191717,
          "tv": 0.201527
        }
      ],
      "Qwen3-4B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00125561,
          "flip": 0.00683315,
          "tv": 0.00674331
        },
        {
          "family": "camada",
          "config": "lq3_0",
          "kl": 0.00645469,
          "flip": 0.0139352,
          "tv": 0.013537
        },
        {
          "family": "camada",
          "config": "lq3_29",
          "kl": 0.00767179,
          "flip": 0.0169633,
          "tv": 0.0169015
        },
        {
          "family": "camada",
          "config": "lq3_30",
          "kl": 0.00939687,
          "flip": 0.0198321,
          "tv": 0.0191795
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00949197,
          "flip": 0.0197325,
          "tv": 0.0187226
        },
        {
          "family": "camada",
          "config": "lq3_31",
          "kl": 0.0102514,
          "flip": 0.0194336,
          "tv": 0.019093
        },
        {
          "family": "camada",
          "config": "lq3_27",
          "kl": 0.0104028,
          "flip": 0.0204098,
          "tv": 0.0201681
        },
        {
          "family": "camada",
          "config": "lq3_26",
          "kl": 0.0104718,
          "flip": 0.0209178,
          "tv": 0.020127
        },
        {
          "family": "camada",
          "config": "lq3_32",
          "kl": 0.010578,
          "flip": 0.0205293,
          "tv": 0.0201903
        },
        {
          "family": "camada",
          "config": "lq3_33",
          "kl": 0.010882,
          "flip": 0.0200612,
          "tv": 0.0196497
        },
        {
          "family": "camada",
          "config": "lq3_1",
          "kl": 0.0112258,
          "flip": 0.0181288,
          "tv": 0.0179309
        },
        {
          "family": "camada",
          "config": "lq3_34",
          "kl": 0.0113795,
          "flip": 0.0216848,
          "tv": 0.0216517
        },
        {
          "family": "camada",
          "config": "lq3_28",
          "kl": 0.0122035,
          "flip": 0.0216449,
          "tv": 0.0208086
        },
        {
          "family": "camada",
          "config": "lq3_25",
          "kl": 0.0126245,
          "flip": 0.0231291,
          "tv": 0.0218597
        },
        {
          "family": "camada",
          "config": "lq3_4",
          "kl": 0.0142922,
          "flip": 0.0213461,
          "tv": 0.0208627
        },
        {
          "family": "camada",
          "config": "lq3_8",
          "kl": 0.0145355,
          "flip": 0.0216649,
          "tv": 0.0210408
        },
        {
          "family": "camada",
          "config": "lq3_2",
          "kl": 0.0163188,
          "flip": 0.0225613,
          "tv": 0.0215616
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.016402,
          "flip": 0.023926,
          "tv": 0.023181
        },
        {
          "family": "camada",
          "config": "lq3_3",
          "kl": 0.016627,
          "flip": 0.0217147,
          "tv": 0.0210378
        },
        {
          "family": "camada",
          "config": "lq3_9",
          "kl": 0.016653,
          "flip": 0.0234578,
          "tv": 0.0230087
        },
        {
          "family": "camada",
          "config": "lq3_7",
          "kl": 0.0178204,
          "flip": 0.0245435,
          "tv": 0.0233715
        },
        {
          "family": "camada",
          "config": "lq3_11",
          "kl": 0.0189874,
          "flip": 0.0254699,
          "tv": 0.024708
        },
        {
          "family": "camada",
          "config": "lq3_10",
          "kl": 0.0190416,
          "flip": 0.0248623,
          "tv": 0.0243187
        },
        {
          "family": "camada",
          "config": "lq3_24",
          "kl": 0.0191522,
          "flip": 0.0272529,
          "tv": 0.0269786
        },
        {
          "family": "camada",
          "config": "lq3_12",
          "kl": 0.0215442,
          "flip": 0.0268744,
          "tv": 0.0260741
        },
        {
          "family": "camada",
          "config": "lq3_13",
          "kl": 0.0217235,
          "flip": 0.0274621,
          "tv": 0.0268311
        },
        {
          "family": "camada",
          "config": "lq3_5",
          "kl": 0.0217645,
          "flip": 0.0258185,
          "tv": 0.0252591
        },
        {
          "family": "camada",
          "config": "lq3_35",
          "kl": 0.0221087,
          "flip": 0.0330501,
          "tv": 0.0311149
        },
        {
          "family": "camada",
          "config": "lq3_14",
          "kl": 0.0225028,
          "flip": 0.0277609,
          "tv": 0.0275104
        },
        {
          "family": "camada",
          "config": "lq3_23",
          "kl": 0.0235499,
          "flip": 0.030291,
          "tv": 0.0291783
        },
        {
          "family": "camada",
          "config": "lq3_17",
          "kl": 0.0241205,
          "flip": 0.0293945,
          "tv": 0.0285764
        },
        {
          "family": "camada",
          "config": "lq3_16",
          "kl": 0.0254868,
          "flip": 0.0290458,
          "tv": 0.0293473
        },
        {
          "family": "camada",
          "config": "lq3_20",
          "kl": 0.0261709,
          "flip": 0.0307492,
          "tv": 0.0301453
        },
        {
          "family": "camada",
          "config": "lq3_22",
          "kl": 0.0265985,
          "flip": 0.0318249,
          "tv": 0.0313381
        },
        {
          "family": "camada",
          "config": "lq3_15",
          "kl": 0.0285075,
          "flip": 0.030779,
          "tv": 0.0300261
        },
        {
          "family": "camada",
          "config": "lq3_18",
          "kl": 0.0315934,
          "flip": 0.0321437,
          "tv": 0.0317283
        },
        {
          "family": "camada",
          "config": "lq3_21",
          "kl": 0.0319092,
          "flip": 0.0350024,
          "tv": 0.0336206
        },
        {
          "family": "camada",
          "config": "lq3_19",
          "kl": 0.0339704,
          "flip": 0.0340163,
          "tv": 0.0330488
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0375566,
          "flip": 0.0392159,
          "tv": 0.0380382
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0712885,
          "flip": 0.050123,
          "tv": 0.0488346
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0762697,
          "flip": 0.054715,
          "tv": 0.0538762
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0790604,
          "flip": 0.0558007,
          "tv": 0.0544846
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0853866,
          "flip": 0.0594065,
          "tv": 0.0578817
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.133213,
          "flip": 0.0745072,
          "tv": 0.072942
        },
        {
          "family": "skip",
          "config": "skip18",
          "kl": 0.138788,
          "flip": 0.0740689,
          "tv": 0.0733568
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.139325,
          "flip": 0.0755033,
          "tv": 0.0741206
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.140581,
          "flip": 0.0771169,
          "tv": 0.0759479
        },
        {
          "family": "camada",
          "config": "lq3_6",
          "kl": 0.154228,
          "flip": 0.0766488,
          "tv": 0.0749889
        },
        {
          "family": "skip",
          "config": "skip32",
          "kl": 0.164031,
          "flip": 0.0780134,
          "tv": 0.0782673
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.335983,
          "flip": 0.124919,
          "tv": 0.126697
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.39796,
          "flip": 0.130696,
          "tv": 0.131611
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.403176,
          "flip": 0.135816,
          "tv": 0.136788
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.45586,
          "flip": 0.150837,
          "tv": 0.151556
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.475664,
          "flip": 0.156644,
          "tv": 0.159326
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.520843,
          "flip": 0.167093,
          "tv": 0.172458
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.670057,
          "flip": 0.186258,
          "tv": 0.191932
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.689559,
          "flip": 0.197972,
          "tv": 0.206776
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 1.21165,
          "flip": 0.269501,
          "tv": 0.281567
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 1.66887,
          "flip": 0.294532,
          "tv": 0.29968
        }
      ],
      "Qwen3-8B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000895986,
          "flip": 0.00685677,
          "tv": 0.00628841
        },
        {
          "family": "oficial",
          "config": "ofc@Qwen/Qwen3-8B-AWQ",
          "kl": 0.0369606,
          "flip": 0.0442208,
          "tv": 0.0410996
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0397321,
          "flip": 0.0454175,
          "tv": 0.0420391
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0557064,
          "flip": 0.0531866,
          "tv": 0.0505022
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0891532,
          "flip": 0.0671356,
          "tv": 0.0633837
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.101186,
          "flip": 0.0685187,
          "tv": 0.0645676
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.509909,
          "flip": 0.170193,
          "tv": 0.169575
        }
      ],
      "gemma-3-12b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000906019,
          "flip": 0.00678876,
          "tv": 0.00608982
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0056968,
          "flip": 0.0159221,
          "tv": 0.0153196
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0121516,
          "flip": 0.0236557,
          "tv": 0.0219831
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0221999,
          "flip": 0.0313776,
          "tv": 0.0298936
        },
        {
          "family": "skip",
          "config": "skip44",
          "kl": 0.0332501,
          "flip": 0.0344454,
          "tv": 0.0333379
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0494948,
          "flip": 0.0489677,
          "tv": 0.0467375
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0566053,
          "flip": 0.0510206,
          "tv": 0.0490221
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0699445,
          "flip": 0.0563163,
          "tv": 0.0540491
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0790139,
          "flip": 0.0623352,
          "tv": 0.0602914
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0837413,
          "flip": 0.0633617,
          "tv": 0.061278
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0848768,
          "flip": 0.0633034,
          "tv": 0.0598617
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.087472,
          "flip": 0.0636417,
          "tv": 0.0606412
        },
        {
          "family": "skip",
          "config": "skip22",
          "kl": 0.180434,
          "flip": 0.0946693,
          "tv": 0.094384
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.231539,
          "flip": 0.111979,
          "tv": 0.110764
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.258793,
          "flip": 0.118442,
          "tv": 0.125213
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.334853,
          "flip": 0.139636,
          "tv": 0.144472
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.341583,
          "flip": 0.140056,
          "tv": 0.149138
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.37212,
          "flip": 0.141596,
          "tv": 0.142246
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.443226,
          "flip": 0.16349,
          "tv": 0.187346
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.488836,
          "flip": 0.168704,
          "tv": 0.179391
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.621518,
          "flip": 0.175283,
          "tv": 0.177681
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 1.03955,
          "flip": 0.265695,
          "tv": 0.288615
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 1.1725,
          "flip": 0.248618,
          "tv": 0.259373
        }
      ],
      "gemma-3-1b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00278765,
          "flip": 0.0103878,
          "tv": 0.0103045
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0286215,
          "flip": 0.035871,
          "tv": 0.0331307
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0485387,
          "flip": 0.0430098,
          "tv": 0.0409298
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.104061,
          "flip": 0.066968,
          "tv": 0.0633987
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.219862,
          "flip": 0.10201,
          "tv": 0.0985409
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.259102,
          "flip": 0.112409,
          "tv": 0.107935
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.278802,
          "flip": 0.11852,
          "tv": 0.117122
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.366424,
          "flip": 0.137019,
          "tv": 0.137486
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.457582,
          "flip": 0.141196,
          "tv": 0.138426
        },
        {
          "family": "skip",
          "config": "skip13",
          "kl": 0.599563,
          "flip": 0.185599,
          "tv": 0.184549
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.880131,
          "flip": 0.245759,
          "tv": 0.26154
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.913841,
          "flip": 0.248411,
          "tv": 0.262474
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 1.66312,
          "flip": 0.380126,
          "tv": 0.409839
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 2.1451,
          "flip": 0.440652,
          "tv": 0.485302
        }
      ],
      "gemma-3-27b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000728522,
          "flip": 0.00537108,
          "tv": 0.00496926
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00812321,
          "flip": 0.0168327,
          "tv": 0.0165287
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.01577,
          "flip": 0.0236427,
          "tv": 0.0234489
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0365671,
          "flip": 0.0376471,
          "tv": 0.035529
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0503835,
          "flip": 0.0440106,
          "tv": 0.0432759
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0587923,
          "flip": 0.0471613,
          "tv": 0.0456003
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0614909,
          "flip": 0.0467023,
          "tv": 0.0461347
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0668401,
          "flip": 0.0482901,
          "tv": 0.0475836
        },
        {
          "family": "skip",
          "config": "skip30",
          "kl": 0.11666,
          "flip": 0.0625178,
          "tv": 0.0608718
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.291697,
          "flip": 0.11577,
          "tv": 0.118531
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.845157,
          "flip": 0.210253,
          "tv": 0.222303
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.944961,
          "flip": 0.224283,
          "tv": 0.235218
        }
      ],
      "gemma-3-4b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00161424,
          "flip": 0.00768135,
          "tv": 0.00773807
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0135448,
          "flip": 0.0240832,
          "tv": 0.0221356
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0255434,
          "flip": 0.0322895,
          "tv": 0.0306503
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0520501,
          "flip": 0.045831,
          "tv": 0.043984
        },
        {
          "family": "skip",
          "config": "skip30",
          "kl": 0.0805455,
          "flip": 0.0503519,
          "tv": 0.0496317
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.108501,
          "flip": 0.0658646,
          "tv": 0.0643853
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.122023,
          "flip": 0.0739531,
          "tv": 0.0712419
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.125226,
          "flip": 0.0751851,
          "tv": 0.0725874
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.161623,
          "flip": 0.0838092,
          "tv": 0.0808594
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.175082,
          "flip": 0.0867339,
          "tv": 0.0845533
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.180594,
          "flip": 0.0901407,
          "tv": 0.0896258
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.181385,
          "flip": 0.088748,
          "tv": 0.0870033
        },
        {
          "family": "skip",
          "config": "skip17",
          "kl": 0.308085,
          "flip": 0.117491,
          "tv": 0.115991
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.467367,
          "flip": 0.156262,
          "tv": 0.156986
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.499438,
          "flip": 0.172964,
          "tv": 0.193228
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.566633,
          "flip": 0.177646,
          "tv": 0.182251
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.720008,
          "flip": 0.201076,
          "tv": 0.20933
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.725756,
          "flip": 0.2097,
          "tv": 0.220255
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.808079,
          "flip": 0.206614,
          "tv": 0.213863
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.851284,
          "flip": 0.184749,
          "tv": 0.184358
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.946099,
          "flip": 0.223713,
          "tv": 0.227244
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 1.35929,
          "flip": 0.294366,
          "tv": 0.314512
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 1.43456,
          "flip": 0.331819,
          "tv": 0.367037
        }
      ],
      "phi-4": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000339602,
          "flip": 0.00421613,
          "tv": 0.00401034
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00103427,
          "flip": 0.00830127,
          "tv": 0.00782567
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00216265,
          "flip": 0.0120344,
          "tv": 0.0111632
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.00329858,
          "flip": 0.0154318,
          "tv": 0.014085
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.00865675,
          "flip": 0.0241097,
          "tv": 0.0223745
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0122021,
          "flip": 0.0309783,
          "tv": 0.0276545
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0129326,
          "flip": 0.0303889,
          "tv": 0.0281565
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0130363,
          "flip": 0.030176,
          "tv": 0.0282394
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0133442,
          "flip": 0.0306508,
          "tv": 0.0280841
        },
        {
          "family": "gauss",
          "config": "s1g4",
          "kl": 0.0133518,
          "flip": 0.0307982,
          "tv": 0.0282632
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0139289,
          "flip": 0.0317069,
          "tv": 0.0296272
        },
        {
          "family": "skip",
          "config": "skip20",
          "kl": 0.0217517,
          "flip": 0.039034,
          "tv": 0.0363812
        },
        {
          "family": "skip",
          "config": "skip36",
          "kl": 0.0361254,
          "flip": 0.0433893,
          "tv": 0.0394322
        },
        {
          "family": "awq",
          "config": "awq3",
          "kl": 0.0596903,
          "flip": 0.065133,
          "tv": 0.0614303
        },
        {
          "family": "kv",
          "config": "kv2",
          "kl": 0.0650764,
          "flip": 0.0652886,
          "tv": 0.0675498
        },
        {
          "family": "rtn",
          "config": "u3",
          "kl": 0.066036,
          "flip": 0.0680639,
          "tv": 0.064248
        },
        {
          "family": "gauss",
          "config": "g3",
          "kl": 0.0680882,
          "flip": 0.0697094,
          "tv": 0.0680584
        },
        {
          "family": "gptq",
          "config": "gptq3",
          "kl": 0.073183,
          "flip": 0.073115,
          "tv": 0.068328
        },
        {
          "family": "magnitude",
          "config": "mag40",
          "kl": 0.0869308,
          "flip": 0.0802702,
          "tv": 0.0752307
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.121331,
          "flip": 0.0933934,
          "tv": 0.0940235
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.137232,
          "flip": 0.100573,
          "tv": 0.100677
        },
        {
          "family": "wanda",
          "config": "wanda24",
          "kl": 0.287878,
          "flip": 0.144585,
          "tv": 0.150256
        },
        {
          "family": "sparsegpt",
          "config": "sgpt24",
          "kl": 0.429987,
          "flip": 0.176267,
          "tv": 0.200314
        }
      ]
    },
    "mmlu_en": {
      "Meta-Llama-3-8B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000863208,
          "flip": 0.00877193,
          "tv": 0.00875393
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00560835,
          "flip": 0.0253376,
          "tv": 0.0227598
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0133835,
          "flip": 0.0378644,
          "tv": 0.034141
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0206423,
          "flip": 0.0482141,
          "tv": 0.0434422
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.053466,
          "flip": 0.0745614,
          "tv": 0.0700977
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0545944,
          "flip": 0.0767702,
          "tv": 0.0722146
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0587776,
          "flip": 0.0786318,
          "tv": 0.073438
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.061786,
          "flip": 0.0851319,
          "tv": 0.0782245
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0893105,
          "flip": 0.0949767,
          "tv": 0.0906451
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0965961,
          "flip": 0.100341,
          "tv": 0.0947542
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.332507,
          "flip": 0.195033,
          "tv": 0.188294
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.369157,
          "flip": 0.19355,
          "tv": 0.189378
        }
      ],
      "Mistral-7B-Instruct-v0.3": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00084537,
          "flip": 0.00964736,
          "tv": 0.00848836
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00576399,
          "flip": 0.0235853,
          "tv": 0.0221455
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0141095,
          "flip": 0.0390718,
          "tv": 0.0339172
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0278058,
          "flip": 0.0526289,
          "tv": 0.0480634
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0419393,
          "flip": 0.0656528,
          "tv": 0.0615812
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.057217,
          "flip": 0.075808,
          "tv": 0.0694667
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0630613,
          "flip": 0.0780421,
          "tv": 0.0724635
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0653502,
          "flip": 0.0796415,
          "tv": 0.0741095
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0930491,
          "flip": 0.0925893,
          "tv": 0.088257
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.103933,
          "flip": 0.0995963,
          "tv": 0.0940906
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.284731,
          "flip": 0.163421,
          "tv": 0.157459
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.288509,
          "flip": 0.167839,
          "tv": 0.159381
        }
      ],
      "OLMo-2-0425-1B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000800198,
          "flip": 0.0129798,
          "tv": 0.0107994
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00671941,
          "flip": 0.0363392,
          "tv": 0.0308743
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.015132,
          "flip": 0.0558196,
          "tv": 0.0458065
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0269161,
          "flip": 0.074213,
          "tv": 0.062195
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0813392,
          "flip": 0.121593,
          "tv": 0.10414
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0830617,
          "flip": 0.127624,
          "tv": 0.107928
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0943369,
          "flip": 0.133699,
          "tv": 0.114481
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0999066,
          "flip": 0.133187,
          "tv": 0.116034
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.116005,
          "flip": 0.150664,
          "tv": 0.129926
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.122421,
          "flip": 0.149961,
          "tv": 0.128555
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.393452,
          "flip": 0.259021,
          "tv": 0.233353
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.44809,
          "flip": 0.270317,
          "tv": 0.243119
        }
      ],
      "Phi-3.5-mini-instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00311129,
          "flip": 0.0151995,
          "tv": 0.01507
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0107179,
          "flip": 0.0331366,
          "tv": 0.0316782
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0150113,
          "flip": 0.0378133,
          "tv": 0.0360622
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0318134,
          "flip": 0.0544957,
          "tv": 0.0524056
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0638198,
          "flip": 0.0768529,
          "tv": 0.0725902
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0704022,
          "flip": 0.0838965,
          "tv": 0.0785775
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0944098,
          "flip": 0.0882596,
          "tv": 0.0842474
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.137722,
          "flip": 0.112071,
          "tv": 0.105491
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.145548,
          "flip": 0.115065,
          "tv": 0.114024
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.151089,
          "flip": 0.110417,
          "tv": 0.105682
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.41386,
          "flip": 0.19996,
          "tv": 0.199386
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.512739,
          "flip": 0.232355,
          "tv": 0.239481
        }
      ],
      "Qwen3-14B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00130303,
          "flip": 0.00945752,
          "tv": 0.00860028
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00746189,
          "flip": 0.0230111,
          "tv": 0.0211971
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0128434,
          "flip": 0.0275733,
          "tv": 0.0270285
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0284477,
          "flip": 0.0456226,
          "tv": 0.0424915
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0546549,
          "flip": 0.060708,
          "tv": 0.0567852
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0552391,
          "flip": 0.0608412,
          "tv": 0.057541
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0700392,
          "flip": 0.0728629,
          "tv": 0.0662724
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0755593,
          "flip": 0.071431,
          "tv": 0.0670269
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.114382,
          "flip": 0.0906124,
          "tv": 0.0882219
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.119706,
          "flip": 0.0909454,
          "tv": 0.0896221
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.339739,
          "flip": 0.154184,
          "tv": 0.152276
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.377236,
          "flip": 0.154717,
          "tv": 0.150946
        }
      ],
      "Qwen3-4B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00249014,
          "flip": 0.0105552,
          "tv": 0.00992579
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0183897,
          "flip": 0.0285069,
          "tv": 0.0273116
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0351973,
          "flip": 0.0376239,
          "tv": 0.0365446
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0703489,
          "flip": 0.0587858,
          "tv": 0.0564599
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.149102,
          "flip": 0.084519,
          "tv": 0.0820955
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.15109,
          "flip": 0.0782526,
          "tv": 0.076348
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.164277,
          "flip": 0.0842622,
          "tv": 0.0832761
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.165637,
          "flip": 0.0823874,
          "tv": 0.0816784
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.279181,
          "flip": 0.115029,
          "tv": 0.111411
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.305727,
          "flip": 0.11223,
          "tv": 0.110075
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.617078,
          "flip": 0.193282,
          "tv": 0.195749
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.634456,
          "flip": 0.192871,
          "tv": 0.194785
        }
      ],
      "gemma-3-12b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.0020278,
          "flip": 0.0106441,
          "tv": 0.0107824
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0124871,
          "flip": 0.0268083,
          "tv": 0.0263861
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0285651,
          "flip": 0.0400515,
          "tv": 0.0389405
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0482925,
          "flip": 0.0543839,
          "tv": 0.0515219
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.113183,
          "flip": 0.0824051,
          "tv": 0.0801804
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.121991,
          "flip": 0.086935,
          "tv": 0.0829313
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.145924,
          "flip": 0.0968365,
          "tv": 0.0945291
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.158056,
          "flip": 0.0977029,
          "tv": 0.0949144
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.187999,
          "flip": 0.107654,
          "tv": 0.104708
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.201222,
          "flip": 0.108594,
          "tv": 0.105009
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.661216,
          "flip": 0.239492,
          "tv": 0.26366
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.835354,
          "flip": 0.25375,
          "tv": 0.269111
        }
      ],
      "gemma-3-1b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00484193,
          "flip": 0.0174591,
          "tv": 0.0167324
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0530922,
          "flip": 0.0583872,
          "tv": 0.0537153
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0853261,
          "flip": 0.0718524,
          "tv": 0.0667249
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.180505,
          "flip": 0.105249,
          "tv": 0.100243
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.320208,
          "flip": 0.144922,
          "tv": 0.139194
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.4497,
          "flip": 0.178851,
          "tv": 0.173674
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.45709,
          "flip": 0.188741,
          "tv": 0.185491
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.531493,
          "flip": 0.18741,
          "tv": 0.184427
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.662155,
          "flip": 0.215938,
          "tv": 0.213856
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.686379,
          "flip": 0.212704,
          "tv": 0.210384
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 1.11732,
          "flip": 0.333054,
          "tv": 0.350785
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 1.2145,
          "flip": 0.34496,
          "tv": 0.362612
        }
      ],
      "gemma-3-4b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.003656,
          "flip": 0.0137976,
          "tv": 0.0133043
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0320026,
          "flip": 0.0391226,
          "tv": 0.0378142
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0573091,
          "flip": 0.0523895,
          "tv": 0.0511972
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.115826,
          "flip": 0.0765353,
          "tv": 0.0735622
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.236151,
          "flip": 0.109142,
          "tv": 0.105937
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.265337,
          "flip": 0.122203,
          "tv": 0.118147
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.281918,
          "flip": 0.130134,
          "tv": 0.125597
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.318638,
          "flip": 0.133288,
          "tv": 0.128757
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.363291,
          "flip": 0.145081,
          "tv": 0.139466
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.364982,
          "flip": 0.14458,
          "tv": 0.143608
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 1.37163,
          "flip": 0.299213,
          "tv": 0.300162
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 1.57569,
          "flip": 0.27377,
          "tv": 0.270787
        }
      ]
    },
    "wikitext": {
      "Meta-Llama-3-8B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.0011551,
          "flip": 0.0130997,
          "tv": 0.0117642
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00771881,
          "flip": 0.0362512,
          "tv": 0.0304291
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0160072,
          "flip": 0.0514902,
          "tv": 0.0428745
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.029463,
          "flip": 0.0688685,
          "tv": 0.0588007
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0681282,
          "flip": 0.103596,
          "tv": 0.0940199
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0690802,
          "flip": 0.0980863,
          "tv": 0.0874266
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0712622,
          "flip": 0.103508,
          "tv": 0.0920373
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0743504,
          "flip": 0.108373,
          "tv": 0.0940533
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.129655,
          "flip": 0.139906,
          "tv": 0.125381
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.130602,
          "flip": 0.140375,
          "tv": 0.12471
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.391724,
          "flip": 0.239018,
          "tv": 0.226398
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.481669,
          "flip": 0.261466,
          "tv": 0.248839
        }
      ],
      "Mistral-7B-Instruct-v0.3": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000989032,
          "flip": 0.0115934,
          "tv": 0.0102933
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00738585,
          "flip": 0.0324244,
          "tv": 0.0278694
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.01555,
          "flip": 0.0452606,
          "tv": 0.039799
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0302163,
          "flip": 0.0652755,
          "tv": 0.0561292
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0476716,
          "flip": 0.0826006,
          "tv": 0.0727639
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0674161,
          "flip": 0.0938972,
          "tv": 0.0841185
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.068856,
          "flip": 0.0941013,
          "tv": 0.0849495
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0730812,
          "flip": 0.0962716,
          "tv": 0.0887413
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.114085,
          "flip": 0.124133,
          "tv": 0.110743
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.122786,
          "flip": 0.126767,
          "tv": 0.115646
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.311208,
          "flip": 0.202078,
          "tv": 0.189988
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.345065,
          "flip": 0.214246,
          "tv": 0.205423
        }
      ],
      "OLMo-2-0425-1B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000896695,
          "flip": 0.0167196,
          "tv": 0.0128677
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00734597,
          "flip": 0.048389,
          "tv": 0.0363542
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0155883,
          "flip": 0.07169,
          "tv": 0.0525399
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0287807,
          "flip": 0.0935476,
          "tv": 0.0717379
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0837956,
          "flip": 0.157746,
          "tv": 0.122496
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0857201,
          "flip": 0.15857,
          "tv": 0.123918
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0947672,
          "flip": 0.161062,
          "tv": 0.126574
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.113839,
          "flip": 0.176407,
          "tv": 0.139808
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.119145,
          "flip": 0.185549,
          "tv": 0.148093
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.12935,
          "flip": 0.190446,
          "tv": 0.150528
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.396923,
          "flip": 0.316041,
          "tv": 0.264346
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.434559,
          "flip": 0.327691,
          "tv": 0.273057
        }
      ],
      "OLMo-2-1124-7B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000638415,
          "flip": 0.0126714,
          "tv": 0.0101217
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00479964,
          "flip": 0.0342342,
          "tv": 0.0273441
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0122933,
          "flip": 0.0546808,
          "tv": 0.0432778
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0184415,
          "flip": 0.0674305,
          "tv": 0.0533013
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0605052,
          "flip": 0.116941,
          "tv": 0.0958993
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0619103,
          "flip": 0.116745,
          "tv": 0.0972204
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0621919,
          "flip": 0.116784,
          "tv": 0.0978522
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0775553,
          "flip": 0.131688,
          "tv": 0.110631
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.081538,
          "flip": 0.133529,
          "tv": 0.111262
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.120067,
          "flip": 0.142636,
          "tv": 0.128849
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.361156,
          "flip": 0.264101,
          "tv": 0.233259
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.399084,
          "flip": 0.269271,
          "tv": 0.239034
        }
      ],
      "Phi-3.5-mini-instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00339301,
          "flip": 0.0228615,
          "tv": 0.0199253
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0124314,
          "flip": 0.0502647,
          "tv": 0.043243
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0145113,
          "flip": 0.0544687,
          "tv": 0.0446503
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0394463,
          "flip": 0.089697,
          "tv": 0.0731074
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0566247,
          "flip": 0.104918,
          "tv": 0.0873143
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0684315,
          "flip": 0.117898,
          "tv": 0.0981728
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0733486,
          "flip": 0.120015,
          "tv": 0.0992742
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.149521,
          "flip": 0.16267,
          "tv": 0.137573
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.161573,
          "flip": 0.173364,
          "tv": 0.145849
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.164724,
          "flip": 0.171646,
          "tv": 0.150753
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.466875,
          "flip": 0.295175,
          "tv": 0.270194
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.47362,
          "flip": 0.274553,
          "tv": 0.244509
        }
      ],
      "Qwen3-14B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00179271,
          "flip": 0.0123653,
          "tv": 0.0123326
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0125827,
          "flip": 0.0354544,
          "tv": 0.0331107
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0178737,
          "flip": 0.0414182,
          "tv": 0.0379945
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0524334,
          "flip": 0.0706899,
          "tv": 0.0667216
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0785077,
          "flip": 0.0861739,
          "tv": 0.0807403
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0822142,
          "flip": 0.0857635,
          "tv": 0.0819022
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0873392,
          "flip": 0.0896482,
          "tv": 0.0865604
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0976988,
          "flip": 0.0991136,
          "tv": 0.0907636
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.238582,
          "flip": 0.14655,
          "tv": 0.13625
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.241474,
          "flip": 0.150161,
          "tv": 0.141132
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.420025,
          "flip": 0.203152,
          "tv": 0.194446
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.537636,
          "flip": 0.221836,
          "tv": 0.212141
        }
      ],
      "Qwen3-4B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00343111,
          "flip": 0.0158711,
          "tv": 0.0155724
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0321851,
          "flip": 0.0477351,
          "tv": 0.0455206
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0460162,
          "flip": 0.0593414,
          "tv": 0.0549109
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.107483,
          "flip": 0.0897432,
          "tv": 0.0862037
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.177562,
          "flip": 0.118378,
          "tv": 0.111586
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.20148,
          "flip": 0.125476,
          "tv": 0.116281
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.225066,
          "flip": 0.134584,
          "tv": 0.126103
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.235066,
          "flip": 0.13696,
          "tv": 0.129215
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.398724,
          "flip": 0.185944,
          "tv": 0.176645
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.461101,
          "flip": 0.188016,
          "tv": 0.180917
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.775514,
          "flip": 0.263472,
          "tv": 0.257162
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.9733,
          "flip": 0.285131,
          "tv": 0.277755
        }
      ],
      "gemma-3-12b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00340199,
          "flip": 0.01741,
          "tv": 0.0162792
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.021587,
          "flip": 0.0446216,
          "tv": 0.0406192
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0440464,
          "flip": 0.0617574,
          "tv": 0.0568165
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.086239,
          "flip": 0.0880552,
          "tv": 0.0796024
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.173117,
          "flip": 0.126348,
          "tv": 0.11681
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.192128,
          "flip": 0.130072,
          "tv": 0.121418
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.193595,
          "flip": 0.133065,
          "tv": 0.124001
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.22201,
          "flip": 0.135853,
          "tv": 0.126878
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.288449,
          "flip": 0.161853,
          "tv": 0.151151
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.291954,
          "flip": 0.161214,
          "tv": 0.150065
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.933877,
          "flip": 0.32277,
          "tv": 0.342139
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 1.27533,
          "flip": 0.353615,
          "tv": 0.35459
        }
      ],
      "gemma-3-1b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00559823,
          "flip": 0.0273946,
          "tv": 0.0239538
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0588265,
          "flip": 0.0844143,
          "tv": 0.0739775
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0961825,
          "flip": 0.1028,
          "tv": 0.0934503
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.189827,
          "flip": 0.151526,
          "tv": 0.138123
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.380441,
          "flip": 0.204821,
          "tv": 0.190551
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.394869,
          "flip": 0.22312,
          "tv": 0.212878
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.403591,
          "flip": 0.220001,
          "tv": 0.207987
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.52477,
          "flip": 0.237499,
          "tv": 0.223229
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.687083,
          "flip": 0.285813,
          "tv": 0.273404
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.751016,
          "flip": 0.288629,
          "tv": 0.275429
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 1.0143,
          "flip": 0.370791,
          "tv": 0.385766
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 1.34469,
          "flip": 0.421963,
          "tv": 0.443751
        }
      ],
      "gemma-3-4b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00415283,
          "flip": 0.0183363,
          "tv": 0.0171008
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0338056,
          "flip": 0.0540094,
          "tv": 0.0481148
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0622595,
          "flip": 0.0729757,
          "tv": 0.0648403
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.112064,
          "flip": 0.100219,
          "tv": 0.0933728
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.242025,
          "flip": 0.138934,
          "tv": 0.130115
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.257051,
          "flip": 0.14606,
          "tv": 0.13899
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.277397,
          "flip": 0.153686,
          "tv": 0.143701
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.35127,
          "flip": 0.165765,
          "tv": 0.155899
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.410272,
          "flip": 0.185166,
          "tv": 0.177459
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.411383,
          "flip": 0.187512,
          "tv": 0.178394
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 1.22004,
          "flip": 0.304982,
          "tv": 0.299027
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 1.24385,
          "flip": 0.312911,
          "tv": 0.309167
        }
      ]
    },
    "wikitext_nat": {
      "Meta-Llama-3-8B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00106339,
          "flip": 0.0139866,
          "tv": 0.0128343
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00585272,
          "flip": 0.0362829,
          "tv": 0.0297673
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0107515,
          "flip": 0.0479863,
          "tv": 0.0407919
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0223623,
          "flip": 0.0668888,
          "tv": 0.0576877
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.043874,
          "flip": 0.0914887,
          "tv": 0.0784654
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0543478,
          "flip": 0.10418,
          "tv": 0.0939422
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0554392,
          "flip": 0.105352,
          "tv": 0.0915733
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.10168,
          "flip": 0.140154,
          "tv": 0.125954
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.102034,
          "flip": 0.143301,
          "tv": 0.125337
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.136695,
          "flip": 0.171953,
          "tv": 0.15569
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.302865,
          "flip": 0.233453,
          "tv": 0.225279
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.431263,
          "flip": 0.283414,
          "tv": 0.275971
        }
      ],
      "Mistral-7B-Instruct-v0.3": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000566489,
          "flip": 0.0116501,
          "tv": 0.00937274
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00338515,
          "flip": 0.0277326,
          "tv": 0.0223157
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00740014,
          "flip": 0.040257,
          "tv": 0.0325073
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0138704,
          "flip": 0.0578643,
          "tv": 0.0457047
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.0226666,
          "flip": 0.071995,
          "tv": 0.0591942
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0288686,
          "flip": 0.0808596,
          "tv": 0.0657813
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0293697,
          "flip": 0.0843567,
          "tv": 0.0664428
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0495953,
          "flip": 0.0983857,
          "tv": 0.0835483
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0506703,
          "flip": 0.10971,
          "tv": 0.0868754
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0561772,
          "flip": 0.11272,
          "tv": 0.0914689
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.161333,
          "flip": 0.187968,
          "tv": 0.158341
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.191069,
          "flip": 0.20651,
          "tv": 0.173398
        }
      ],
      "OLMo-2-0425-1B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.0007368,
          "flip": 0.0150146,
          "tv": 0.0115737
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00654375,
          "flip": 0.046875,
          "tv": 0.0335767
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0132995,
          "flip": 0.0606893,
          "tv": 0.0457947
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0273329,
          "flip": 0.0923665,
          "tv": 0.0676203
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0707463,
          "flip": 0.138753,
          "tv": 0.107127
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0730034,
          "flip": 0.145711,
          "tv": 0.112179
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.113645,
          "flip": 0.182983,
          "tv": 0.14446
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.115169,
          "flip": 0.181213,
          "tv": 0.142059
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.120244,
          "flip": 0.184245,
          "tv": 0.145474
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.131274,
          "flip": 0.187703,
          "tv": 0.149279
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.337121,
          "flip": 0.281352,
          "tv": 0.244809
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.459569,
          "flip": 0.334839,
          "tv": 0.295433
        }
      ],
      "OLMo-2-0425-1B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000933117,
          "flip": 0.0177612,
          "tv": 0.0132357
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00799253,
          "flip": 0.052653,
          "tv": 0.0376773
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0187981,
          "flip": 0.0745646,
          "tv": 0.056563
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0306584,
          "flip": 0.0979818,
          "tv": 0.0736267
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0839673,
          "flip": 0.152445,
          "tv": 0.118422
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.1013,
          "flip": 0.167867,
          "tv": 0.132819
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.108031,
          "flip": 0.172221,
          "tv": 0.135657
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.116856,
          "flip": 0.178446,
          "tv": 0.138576
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.133942,
          "flip": 0.19574,
          "tv": 0.154782
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.138633,
          "flip": 0.197449,
          "tv": 0.154445
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.412799,
          "flip": 0.316711,
          "tv": 0.262468
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.508667,
          "flip": 0.355652,
          "tv": 0.300128
        }
      ],
      "OLMo-2-1124-7B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.0003938,
          "flip": 0.0106812,
          "tv": 0.00772052
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.00318298,
          "flip": 0.0290324,
          "tv": 0.0215643
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.00876392,
          "flip": 0.0446981,
          "tv": 0.0346448
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0124989,
          "flip": 0.0559489,
          "tv": 0.0430795
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0434663,
          "flip": 0.100952,
          "tv": 0.0795783
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.0476021,
          "flip": 0.0960083,
          "tv": 0.0780575
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0505171,
          "flip": 0.0983276,
          "tv": 0.0800325
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.0569853,
          "flip": 0.110311,
          "tv": 0.0917094
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.0621725,
          "flip": 0.115682,
          "tv": 0.0923916
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.166988,
          "flip": 0.147563,
          "tv": 0.133584
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.271575,
          "flip": 0.240092,
          "tv": 0.207375
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.309904,
          "flip": 0.237345,
          "tv": 0.201151
        }
      ],
      "OLMo-2-1124-7B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.000882019,
          "flip": 0.0149536,
          "tv": 0.0115459
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0069585,
          "flip": 0.0408325,
          "tv": 0.0319743
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0191488,
          "flip": 0.0647583,
          "tv": 0.0511675
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0288626,
          "flip": 0.0795288,
          "tv": 0.0640978
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.0906345,
          "flip": 0.138977,
          "tv": 0.11555
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0987929,
          "flip": 0.133931,
          "tv": 0.114175
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.104025,
          "flip": 0.133809,
          "tv": 0.114143
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.12327,
          "flip": 0.158081,
          "tv": 0.132089
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.1264,
          "flip": 0.156352,
          "tv": 0.132618
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.204298,
          "flip": 0.169149,
          "tv": 0.150747
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.473337,
          "flip": 0.295227,
          "tv": 0.263595
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.546584,
          "flip": 0.299764,
          "tv": 0.267555
        }
      ],
      "Qwen2.5-72B-Instruct": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00163465,
          "flip": 0.0178833,
          "tv": 0.0151542
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0091674,
          "flip": 0.0403849,
          "tv": 0.0336659
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0128447,
          "flip": 0.0477905,
          "tv": 0.0400211
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0321677,
          "flip": 0.0747884,
          "tv": 0.0625991
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.0517976,
          "flip": 0.0945841,
          "tv": 0.0796173
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.116698,
          "flip": 0.142558,
          "tv": 0.121188
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.139042,
          "flip": 0.162781,
          "tv": 0.137719
        }
      ],
      "Qwen3-4B": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.00252814,
          "flip": 0.020284,
          "tv": 0.0181766
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.0196599,
          "flip": 0.0536702,
          "tv": 0.0477034
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0315371,
          "flip": 0.0692952,
          "tv": 0.0598918
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0714565,
          "flip": 0.0993652,
          "tv": 0.0887636
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.101739,
          "flip": 0.118774,
          "tv": 0.1068
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.115092,
          "flip": 0.132507,
          "tv": 0.119674
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.145133,
          "flip": 0.145386,
          "tv": 0.131439
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.155536,
          "flip": 0.153564,
          "tv": 0.137344
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.266703,
          "flip": 0.197144,
          "tv": 0.180747
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.273788,
          "flip": 0.200317,
          "tv": 0.185483
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 0.430679,
          "flip": 0.236084,
          "tv": 0.219292
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.495304,
          "flip": 0.269104,
          "tv": 0.252785
        }
      ],
      "gemma-3-4b-it": [
        {
          "family": "rtn",
          "config": "u8",
          "kl": 0.0028492,
          "flip": 0.0220133,
          "tv": 0.01894
        },
        {
          "family": "rtn",
          "config": "u6",
          "kl": 0.021027,
          "flip": 0.0588582,
          "tv": 0.0486049
        },
        {
          "family": "kv",
          "config": "kv4",
          "kl": 0.0412818,
          "flip": 0.0829468,
          "tv": 0.0691745
        },
        {
          "family": "rtn",
          "config": "u5",
          "kl": 0.0750026,
          "flip": 0.109965,
          "tv": 0.0933315
        },
        {
          "family": "gptq",
          "config": "gptq4",
          "kl": 0.162519,
          "flip": 0.14919,
          "tv": 0.129569
        },
        {
          "family": "kv",
          "config": "kv3",
          "kl": 0.164241,
          "flip": 0.161296,
          "tv": 0.139942
        },
        {
          "family": "magnitude",
          "config": "mag20",
          "kl": 0.187155,
          "flip": 0.171794,
          "tv": 0.156396
        },
        {
          "family": "awq",
          "config": "awq4",
          "kl": 0.195383,
          "flip": 0.166524,
          "tv": 0.146628
        },
        {
          "family": "rtn",
          "config": "u4",
          "kl": 0.268732,
          "flip": 0.205831,
          "tv": 0.181775
        },
        {
          "family": "gauss",
          "config": "g4",
          "kl": 0.280259,
          "flip": 0.207397,
          "tv": 0.185226
        },
        {
          "family": "wanda",
          "config": "wanda50",
          "kl": 0.891845,
          "flip": 0.34906,
          "tv": 0.324873
        },
        {
          "family": "sparsegpt",
          "config": "sgpt50",
          "kl": 1.06628,
          "flip": 0.354268,
          "tv": 0.331665
        }
      ]
    }
  }
};
