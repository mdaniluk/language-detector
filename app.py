import sys
import uvicorn
from pyhocon import ConfigFactory
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from fastai.basic_train import load_learner


conf = ConfigFactory.parse_file(sys.argv[1])
app = Starlette()
model = load_learner(path= '.', file=conf.model_file_name)


@app.route("/predict", methods=["POST"])
async def predict(request):
    data = await request.json()
    text = data['text']
    output = model.predict(text)
    return JSONResponse(str(output[0]))


if __name__ == "__main__":
    uvicorn.run(app, host = conf.application['host'], port=conf.application['port'])
