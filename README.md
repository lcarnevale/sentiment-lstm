# Twitter Sentiment LSTM

## Microservice

### How to run it
Build the image as follow:
```bash
docker build -t lcarnevale/twitter-sentiment-lstm .
```

Run the container as follow:
```bash
docker run -d -p 5002:5002 --name twitter-sentiment-lstm lcarnevale/twitter-sentiment-lstm
```

Try to send a request out as follow:
```bash
$ curl -d '{"target_question":"I love you"}' -H "Content-Type: application/json" -X POST http://localhost:5002/
```
