# Documentação do Dataset Meteorológico

## Visão Geral

Este projeto utiliza um conjunto de dados meteorológicos históricos do Brasil contendo medições climáticas registradas de hora em hora.

Os dados cobrem o período entre:

- **Ano inicial:** 2000 (primeiro registro: 2000-05-07)
- **Ano final:** 2025 (último registro disponível)

Todos os horários registrados estão em:

- **UTC (Coordinated Universal Time)**

---

# Estrutura dos Dados

O projeto é composto por dois datasets principais:

1. `weather_measurements.parquet`
2. `stations.parquet`

---

# 1. Dataset de Medições Meteorológicas

Arquivo:

```text
weather_measurements.parquet
```

Este dataset contém as observações meteorológicas registradas por estação.

## Frequência Temporal

- 1 registro por hora
- Série histórica contínua
- Dados entre 2000 e 2025

---

## Colunas do Dataset

| Coluna | Tipo | Descrição | Exemplo |
|---|---|---|---|
| `station_code` | `object` (str) | Código alfanumérico da estação | `A001` |
| `measurement_time` | `datetime64[us]` | Data e hora da medição em UTC | `2000-05-07 00:00:00` |
| `temperature` | `float64` | Temperatura do ar | `22.2` |
| `humidity` | `float64` | Umidade relativa do ar | `68.0` |
| `rainfall` | `float64` | Volume de chuva acumulada na hora | `0.0` |
| `global_radiation` | `float64` | Radiação solar global incidente | `2163.31` |
| `pressure` | `float64` | Pressão atmosférica | `886.1` |
| `dew_point` | `float64` | Temperatura do ponto de orvalho | `16.0` |
| `wind_speed` | `float64` | Velocidade média do vento | `1.3` |
| `wind_gust` | `float64` | Rajada máxima de vento | `2.7` |
| `wind_direction` | `float64` | Direção do vento em graus | `101.0` |

---

# Possíveis Unidades Utilizadas

| Variável | Unidade provável |
|---|---|
| `temperature` | °C |
| `humidity` | % |
| `rainfall` | mm |
| `global_radiation` | KJ/m² ou W/m² |
| `pressure` | hPa |
| `dew_point` | °C |
| `wind_speed` | m/s |
| `wind_gust` | m/s |
| `wind_direction` | graus (0–360°) |

---

# Timezone

Todos os registros utilizam o padrão UTC.

## Conversão para Horário de Brasília

Normalmente:

```text
UTC-3
```

Exemplo:

| UTC | Horário Brasília |
|---|---|
| 15:00 UTC | 12:00 BRT |

---

# Observação sobre Valores Ausentes

Os registros mais antigos (início de operação de cada estação) tendem a ter todas as variáveis como `NaN`. Isso é esperado e indica períodos de instalação ou falha do equipamento. Recomenda-se filtrar por períodos com dados válidos antes de qualquer análise.

---

# 2. Dataset de Estações Meteorológicas

Arquivo:

```text
stations.parquet
```

Este dataset contém os metadados das estações meteorológicas.

**Total de estações:** 663

---

## Colunas do Dataset

| Coluna | Tipo | Descrição | Exemplo |
|---|---|---|---|
| `code` | `object` (str) | Código alfanumérico da estação | `A565` |
| `name` | `object` (str) | Nome da estação | `BAMBUI` |
| `altitude` | `float64` | Altitude em metros | `697.0` |
| `latitude` | `object` → `float64` | Latitude geográfica (alta precisão) | `-20.031111` |
| `longitude` | `object` → `float64` | Longitude geográfica (alta precisão) | `-46.008889` |
| `state` | `object` (str) | Sigla do estado brasileiro | `MG` |
| `start_operation` | `datetime64[us]` | Data de início de operação | `2016-11-23` |

> `latitude` e `longitude` são armazenados como strings de alta precisão decimal no parquet e devem ser convertidos para `float64` antes de uso numérico.

---

## Exemplos de Estações

| code | name | altitude (m) | latitude | longitude | state | start_operation |
|---|---|---|---|---|---|---|
| A565 | BAMBUI | 697.00 | -20.031111 | -46.008889 | MG | 2016-11-23 |
| A826 | ALEGRETE | 120.88 | -29.709167 | -55.525556 | RS | 2006-09-28 |
| A924 | ALTA FLORESTA | 291.85 | -10.077222 | -56.179167 | MT | 2007-05-23 |
| A021 | ARAGUAINA | 230.76 | -7.103889 | -48.201111 | TO | 2007-01-26 |
| A502 | BARBACENA | 1168.76 | -21.228333 | -43.767778 | MG | 2002-12-05 |

---

# Relação Entre os Datasets

A ligação entre os datasets ocorre através dos campos:

| Dataset | Campo |
|---|---|
| Medições meteorológicas | `station_code` |
| Estações meteorológicas | `code` |

---

# Possibilidades de Uso

Os dados podem ser utilizados para:

- Análises climáticas históricas
- Séries temporais meteorológicas
- Machine Learning climático
- Modelos de previsão do tempo
- Estudos ambientais
- Monitoramento climático
- Geração de mapas meteorológicos
- Estudos de radiação solar
- Modelagem atmosférica
- Análise de eventos extremos

---

# Considerações Técnicas

## Volume de Dados

Como os dados possuem frequência horária ao longo de mais de 25 anos e 663 estações, o volume total é elevado.

Isso permite:

- análises de longo prazo
- detecção de sazonalidade
- identificação de tendências climáticas
- treinamento de modelos meteorológicos robustos

---

# Observações Importantes

- Os horários estão em UTC
- `latitude` e `longitude` exigem cast para `float64` antes de operações numéricas
- Valores faltantes (`NaN`) são comuns, especialmente no início da série de cada estação
- As unidades podem variar dependendo da origem da estação
- Recomenda-se validação e limpeza dos dados antes de análises avançadas

---

# Resumo

O projeto fornece uma base meteorológica histórica de alta resolução temporal com **663 estações** e dados desde **2000-05-07**, cobrindo todo o território brasileiro, para estudos climáticos, análises ambientais e desenvolvimento de soluções meteorológicas e analíticas.
