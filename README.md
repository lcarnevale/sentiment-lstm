# Tweets Alert Classifier

## How to run it
Build the image as follow:
```bash
docker build -t lcarnevale/tweets-alert-classifier .
```

Run the container as follow:
```bash
docker run -d -p 5002:5002 --name tweets-alert-classifier lcarnevale/tweets-alert-classifier
```

Try to send a request out as follow:
```bash
$ curl -d '{"target_question":"I love you"}' -H "Content-Type: application/json" -X POST http://localhost:5002/
```
