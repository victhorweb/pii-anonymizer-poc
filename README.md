# Anonimizador de currículos (PoC)

Mascara dados pessoais de currículos brasileiros **antes** de qualquer chamada a LLM externa e reidrata a resposta depois. Tudo roda em CPU, localmente.

- **Detecção**: Microsoft Presidio como orquestrador.
  - Regex + **dígito verificador** para CPF e CNPJ (CPF com DV errado é ignorado).
  - Regex para telefone BR (fixo/celular, com/sem DDD, com/sem +55, com/sem formatação), e-mail, CEP, data de nascimento, logradouro e URLs de LinkedIn/GitHub (e outras redes).
  - `GLiNERRecognizer` com `urchade/gliner_multi_pii-v1` (`map_location="cpu"`, threshold **0.3**) para nome, endereço, telefone, e-mail e data de nascimento.
- **Placeholders estáveis**: `<NOME_1>`, `<TELEFONE_2>`, `<CPF_1>`… O mesmo valor no mesmo documento sempre vira o mesmo placeholder, inclusive em grafias diferentes (`MARIA SILVA` / `Maria Silva`, `529.982.247-25` / `52998224725`, `(11) 98765-4321` / `+55 11 98765-4321`).
- **Cidade e estado ficam visíveis**: de qualquer endereço detectado (regex ou GLiNER) sai o sufixo `Cidade/UF`, `Cidade - UF` ou `Cidade, UF`; o resto do endereço (logradouro, número, complemento, bairro) e o CEP continuam mascarados. Um "endereço" que é só cidade ou estado, sem número e sem tipo de logradouro, não é mascarado.
- **Recall primeiro**: toda ocorrência de um valor detectado é propagada pelo texto (se o GLiNER achou o nome uma vez, as outras ocorrências também são mascaradas), e spans sobrepostos são **unidos** num só placeholder: nada fica aninhado e nada detectado vaza.

## Rodar

```bash
./run.sh            # cria .venv na primeira vez, instala e sobe em http://127.0.0.1:8765
# ou
make install && make run
```

A primeira subida baixa o modelo GLiNER (~1,1 GB) do Hugging Face; depois fica em cache. O log mostra `GLiNER pronto em Xs` quando o servidor está pronto.

LLM: com `ANTHROPIC_API_KEY` definida, `/process` chama `claude-sonnet-5-5` pedindo JSON estruturado (structured outputs + `fallbacks: "default"` em caso de recusa). Sem a variável, a resposta é **simulada** ecoando o texto mascarado.

```bash
ANTHROPIC_API_KEY=sk-ant-... ./run.sh
```

## Desempenho

- O GLiNER responde por ~99% do tempo, e o custo cresce com o **tamanho do texto** (~1,5–2 s por mil caracteres no PyTorch, neste notebook). Quantos dados pessoais o texto tem e quantos rótulos o GLiNER procura mexem pouco.
- **Runtime ONNX fp32**: com o export em `models/gliner_multi_pii-v1-onnx/model.onnx`, o GLiNER roda no ONNX Runtime; sem ele, cai para o PyTorch. O log de startup e o `/health` (`ner_runtime`) dizem qual está ativo. Para gerar o export (~1,2 GB, fora do git):

  ```bash
  make export-onnx
  ```

- Pedaços de 600 caracteres com 60 de sobreposição, todos numa única chamada em lote. Com ONNX, isso deixou o currículo de exemplo ~1,7x mais rápido que PyTorch com pedaços de 250/50, com exatamente os mesmos spans detectados.
- `GLINER_THREADS` (padrão 10) controla as threads do ONNX Runtime ou do PyTorch. Neste notebook (i7-1365U, 2 núcleos rápidos + 8 econômicos), 10 foi o melhor; 12 piora bastante. Ajuste para a máquina de produção.
- Cache em memória (LRU, 128 textos) por SHA-256 do texto: "Enviar para LLM" depois de "Mascarar" não roda a detecção de novo.
- O modelo é aquecido no startup com o currículo de exemplo, então a primeira requisição não paga o aquecimento.
- Cada resposta traz `timings` (regex, gliner, llm, total, cache_hit), que também aparecem no log e na tela.
- Quantização int8 (PyTorch e ONNX) foi testada e **descartada**: o modelo deixou de detectar nomes, telefones e e-mails.

## Testes

```bash
make test
```

Cobrem: DV de CPF/CNPJ, CPF válido detectado, CPF inválido ignorado, 10 formatos de telefone, faixas de ano que não viram telefone, nome repetido → mesmo placeholder, spans sem sobreposição, nenhum dado pessoal do exemplo sobrando no texto mascarado e roundtrip `anonymize` → `rehydrate` byte a byte idêntico (direto e via API).

## API

| Método | Rota | Entrada | Saída |
| --- | --- | --- | --- |
| POST | `/anonymize` | `{text}` | `{masked_text, mapping, variants, entities[]}` (cada entidade: `start`, `end`, `text`, `entity_type`, `placeholder`, `score`, `sources`) |
| POST | `/rehydrate` | `{masked_text, mapping, variants?}` | `{text}` |
| POST | `/process` | `{text}` | `{sent_to_llm, llm_response, rehydrated, mapping, entities, simulated, model}` |
| POST | `/extract-text` | arquivo `.txt`/`.pdf` (multipart) | `{filename, text}` |
| GET | `/sample` | – | currículo de exemplo |
| GET | `/health` | – | modo da LLM e modelos |

`mapping` é `placeholder → valor canônico` (a grafia mais frequente) e é o que reidrata a resposta da LLM. `variants` guarda a grafia exata de cada ocorrência quando ela difere da canônica; passando `variants` no `/rehydrate`, o texto original volta idêntico.

## Limitações conhecidas

- O GLiNER é probabilístico: nomes muito incomuns ou fora de contexto podem escapar. O threshold baixo (0.3) troca precisão por recall, então espere alguns falsos positivos (ex.: nome de empresa marcado como pessoa).
- A data de nascimento por regex mascara qualquer data completa (`dd/mm/aaaa`); datas no formato `mm/aaaa`, comuns em experiências, não são tocadas.
- Cidade sem UF no fim de um endereço (`Rua X, 100, Campinas`) continua dentro do placeholder, porque não dá para separá-la do bairro com segurança.
- O mapa de substituição fica só na resposta da API local; numa versão real ele precisa ser guardado com criptografia e ter prazo de retenção.
