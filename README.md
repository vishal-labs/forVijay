docker compose up --build -d 

example curl commands:
```json
curl -X POST http://localhost:8000/predict/combined \
  -H "Content-Type: application/json" \
  -d '{
    "dur": 5.2,
    "proto": "tcp",
    "service": "http",
    "state": "CON",
    "spkts": 80,
    "dpkts": 90,
    "sbytes": 3200,
    "dbytes": 3600,
    "rate": 25.0,
    "sttl": 64,
    "dttl": 64,
    "sload": 100.0,
    "dload": 110.0,
    "ct_dst_src_ltm": 3,
    "ct_state_ttl": 2
  }'
```
can check the /docs for swaggerUI
``` json
curl -X POST http://localhost:8000/predict/combined \
  -H "Content-Type: application/json" \
  -d '{
    "dur": 0.005,
    "proto": "udp",
    "service": "-",
    "state": "INT",
    "spkts": 2000,
    "dpkts": 2,
    "sbytes": 120000,
    "dbytes": 100,
    "rate": 80000,
    "sttl": 255,
    "dttl": 5
  }'
```
