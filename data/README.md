# Data

Everything here is either a public benchmark, copied as released, or numbers transcribed from a paper.
Nothing was edited.

## BBQ (`bbq/`)

Parrish et al. (2022), *BBQ: A Hand-Built Bias Benchmark for Question Answering*.
Source: https://github.com/nyu-mll/BBQ. License: CC-BY-4.0. One jsonl file per category.

| Category | Items | Ambiguous |
|---|---:|---:|
| Age | 3,680 | 1,840 |
| Disability_status | 1,556 | 778 |
| Gender_identity | 5,672 | 2,836 |
| Nationality | 3,080 | 1,540 |
| Physical_appearance | 1,576 | 788 |
| Race_ethnicity | 6,880 | 3,440 |
| Race_x_SES | 11,160 | 5,580 |
| Race_x_gender | 15,960 | 7,980 |
| Religion | 1,200 | 600 |
| SES | 6,864 | 3,432 |
| Sexual_orientation | 864 | 432 |
| **Total** | **58,492** | **29,246** |

Only the ambiguous items are used. The experiments sample 200 per category.

## WinoGender (`winogender/`)

Rudinger et al. (2018), *Gender Bias in Coreference Resolution*.
Source: https://github.com/rudinger/winogender-schemas.

- `all_sentences.tsv`: 720 sentences, 240 each male, female and neutral pronoun. The experiments use the
  480 gendered ones.
- `occupations-stats.tsv`: 60 occupations with the percentage of women in the occupation from Bergsma
  and Lin (2006) and from the US Bureau of Labor Statistics.

## Published BBQ results (`decap_bbq_ambiguous.csv`)

Table 11(a) of Bae, Choi and Lee (2025), *DeCAP*, arXiv:2503.19426. Ambiguous-split accuracy and bias
score for 5 methods on 8 models, 40 rows. Typed in by hand from the paper, in percentage points as printed.
The file header records the source. `decap_source_notes.md` lists the other papers that were checked and
why they could not be used.

## Multilingual item sets

Not copied here. They are built from the public releases and checked in under `experiments/`:

| File | Source | Items |
|---|---|---|
| `multilingual_items.csv` | MBBQ (Neplenbroek et al., 2024, CC-BY-4.0): English, Spanish, Dutch, Turkish; KoBBQ (Jin et al., 2023, MIT): Korean | 1,076 per MBBQ language, 1,140 Korean |
| `cabbq_items.csv` | CaBBQ (BSC-LT, CC-BY-4.0): Catalan, written from scratch | 1,968 across 10 categories |

Items whose stereotyped group can't be matched to one of the answer options are dropped and counted. The
reports (`*_report.json`) have the numbers.

`tests/test_data_files.py` checks the counts above against the files.
