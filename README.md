# IA para Classificação de Instrumentos

Este projeto utiliza técnicas de aprendizado de máquina para classificar diferentes instrumentos musicais a partir de arquivos de áudio. A aplicação é construída com PyQt5 para a interface gráfica e utiliza bibliotecas como TensorFlow, librosa, e seaborn para processamento e visualização de dados.

## Pré-requisitos

Antes de executar o projeto, você precisa instalar as dependências necessárias. Para isso, execute o seguinte comando no terminal:

```bash
pip install -r requirements.txt
```

## Estrutura de Diretórios

Certifique-se de que a estrutura de diretórios do projeto esteja configurada da seguinte forma:

## Configuração dos Dados

1. **Baixe a base de dados**: Você precisará baixar os arquivos de áudio para treino e teste. Coloque os arquivos de áudio na pasta `Train_submission` para os dados de treino e na pasta `Test_submission` para os dados de teste.

2. **Metadados**: Certifique-se de que os arquivos `Metadata_Train.csv` e `Metadata_Test.csv` estejam na pasta `dados/`.

## Executando o Projeto

Após configurar as pastas e instalar as dependências, você pode executar o projeto com o seguinte comando:

```bash
python Desafios_Instrumentos.py
```

## Contribuição

Sinta-se à vontade para contribuir com melhorias ou correções. Para isso, faça um fork do repositório e envie um pull request.

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
