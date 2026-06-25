from fastapi import FastAPI

app= FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/decode/{vin}")
def decode_vin(str):
    if len(str)==17:
        return 
    