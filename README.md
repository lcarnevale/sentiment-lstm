# Sentiment Analysis Using LSTM

This project is part of the scientific research carried out at [University of Messina](https://www.unime.it/en) and it aims to be a research product.

## Microservice

### How to run it
Build the image as follow:
```bash
docker build -t lcarnevale/sentiment-lstm .
```

Run the container as follow:
```bash
docker run -d --rm -p 5002:5002 --name sentiment-lstm lcarnevale/sentiment-lstm
```

### How to use it
Try to send a request out as follow:
```bash
$ curl -d '{ \
  "target_question":"I love you" \
}' \
-H "Content-Type:application/json" \
-X POST http://localhost:5002/
```
