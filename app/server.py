import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import torch
torch.nn.Module.dump_patches = True


# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://drive.google.com/uc?export=download&id=1QzLupiw53K7IRvy52QxwWs5Tnu5zmQUM'
export_file_name = 'export.pkl'

classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
full_classes = ["""Actinic keratoses and intraepithelial carcinoma / Bowen's disease""","""Basal cell carcinoma """,
"""Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)""",
"Dermatofibroma","Melanoma","Melanocytic Nevi","Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)"]
dict_ = {val:key for key,val in zip(full_classes,classes)}
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    # prediction = learn.predict(img)
    # pred = re.findall('(.*?);',str(prediction)+';')
    # print(prediction)
    # pred = prediction[0].obj[-1]
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})
    # out = sorted(zip(learn.data.classes, map(float, prediction[1])),key=lambda p: p[1],reverse=True)
    # # out_ = [dict_[i[0]] for i in out if i[1] == 1.0]
    # out_ = []
    # for i in out:
    # 	if i[1] == 1.0 and i[0] in dict_:
    # 		out_.append(dict_[i[0]])
    # 	out_.append('bcc')

    # return JSONResponse({'result': out_ })


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
