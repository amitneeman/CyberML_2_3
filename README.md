# CyberML_2_3

## Structure

By default, the code is running on port 5000 and ratelimiting access to 10 calls per minute per IP
Single request time is between 2-5 seconds

## Api

```http request
POST /predator/predict 
```
Body: 
```json
{
  "chat": "This is a chat"
}
```

## install

```shell
# Server
pip install flask flask_limiter 
# Model
pip install numpy pandas bert_sklearn
```

### Running the server

```shell
python server.py
```

### Sending request example

```shell
curl  -X POST http://127.0.0.1:5000/predator/predict  \
      -H 'Content-Type: application/json'  \
      -d '{"chat":"This is a chat"}'
      
```