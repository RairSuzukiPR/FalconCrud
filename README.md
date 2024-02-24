# FalconCrud

Simple api to study Falcon Web Framework.

## Passos para Executar

### API

#### Sem Docker:

1. Instale as dependências utilizando pip:
    ```
    pip install torch pandas falcon
    ```
    ```
    pip install scikit-learn
    ```
2. Execute o arquivo `run.py`.


#### Com Docker:

1. Execute o seguinte comando para iniciar o serviço com Docker Compose:
    ```
    sudo docker compose up
    ```

Para informações sobre como os dados do CSV foram gerados, consulte o relatório técnico.


#### Usando rede neural

1. Inicialize a rede neural chamando o endpoint POST http://localhost:8000/init-neural-net com o seguinte corpo (body) em formato JSON:
    ```json
    {
        "input_size": 4,
        "hidden1_size": 8,
        "hidden2_size": 8,
        "output_size": 3,
        "learning_rate": 0.001,
        "epochs": 10000
    }
    ```

2. Treine os dados do CSV contido nos arquivos da API chamando o endpoint PATCH http://localhost:8000/train.
