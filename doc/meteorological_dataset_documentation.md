# Documentação do Dataset Meteorológico

## Visão Geral

Este projeto utiliza um conjunto de dados meteorológicos históricos do Brasil contendo medições climáticas registradas de hora em hora.

Os dados cobrem o período entre:

- **Ano inicial:** 2000
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

| Coluna | Descrição | Exemplo |
|---|---|---|
| `station_code` | Código identificador da estação meteorológica | `A001` |
| `measurement_time` | Data e hora da medição em UTC | `2025-12-25 01:00:00` |
| `temperature` | Temperatura do ar | `22.2` |
| `humidity` | Umidade relativa do ar (%) | `68` |
| `rainfall` | Volume de chuva acumulada na hora | `0` |
| `global_radiation` | Radiação solar global incidente | `2163.31` |
| `pressure` | Pressão atmosférica | `886.1` |
| `dew_point` | Temperatura do ponto de orvalho | `16.0` |
| `wind_speed` | Velocidade média do vento | `1.3` |
| `wind_gust` | Rajada máxima de vento | `2.7` |
| `wind_direction` | Direção do vento em graus | `101` |

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

## Exemplo

```text
2025-12-25 01:00:00 UTC
```

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

# Exemplo de Registro

| Campo | Valor |
|---|---|
| Estação | A001 |
| Horário UTC | 2025-12-25 01:00 |
| Temperatura | 22.2°C |
| Umidade | 68% |
| Chuva | 0 mm |
| Vento | 1.3 m/s |
| Direção do vento | 101° |

---

# 2. Dataset de Estações Meteorológicas

Arquivo:

```text
stations.parquet
```

Este dataset contém os metadados das estações meteorológicas.

---

## Colunas do Dataset

| Coluna | Descrição | Exemplo |
|---|---|---|
| `code` | Código da estação | `A001` |
| `name` | Nome da estação | `GOIANIA` |
| `altitude` | Altitude da estação | `742` |
| `latitude` | Latitude geográfica | `-16.64` |
| `longitude` | Longitude geográfica | `-49.22` |
| `state` | Estado brasileiro | `GO` |
| `start_operation` | Data de início de operação | `2000-01-01` |

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

Como os dados possuem frequência horária ao longo de mais de 25 anos, o volume total é elevado.

Isso permite:

- análises de longo prazo
- detecção de sazonalidade
- identificação de tendências climáticas
- treinamento de modelos meteorológicos robustos

---

# Observações Importantes

- Os horários estão em UTC
- Algumas estações podem possuir períodos sem dados
- Valores faltantes podem existir
- As unidades podem variar dependendo da origem da estação
- Recomenda-se validação e limpeza dos dados antes de análises avançadas

---

# Resumo

O projeto fornece uma base meteorológica histórica de alta resolução temporal para estudos climáticos, análises ambientais e desenvolvimento de soluções meteorológicas e analíticas.
