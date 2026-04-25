import os, json, cv2, numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from face_detection import extract_face

app = FastAPI()
DB_PATH = os.path.join(cv2.data.haarcascades, 'users.json')

def load_db(): return json.load(open(DB_PATH)) if os.path.exists(DB_PATH) else {}
def save_db(d): json.dump(d, open(DB_PATH, 'w'), indent=2)


@app.post('/register')
async def register(name: str = Form(...), image: UploadFile = File(...)):
    try:
        db = load_db()
        if name in db:
            return JSONResponse(status_code=409, content={"success": False, "message": f"'{name}' already registered."})

        face, err = extract_face(await image.read())
        if err:
            return JSONResponse(status_code=422, content={"success": False, "message": err})

        db[name] = face.flatten().tolist()
        save_db(db)
        return {"success": True, "message": f"'{name}' registered!"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Server error: {str(e)}"})


@app.post('/verify')
async def verify(name: str = Form(...), image: UploadFile = File(...)):
    try:
        db = load_db()
        if name not in db:
            return JSONResponse(status_code=404, content={"success": False, "message": f"'{name}' not found. Register first."})

        face, err = extract_face(await image.read())
        if err:
            return JSONResponse(status_code=422, content={"success": False, "message": err})

        stored = np.array(db[name], dtype=np.float32).reshape(100, 100)
        score  = float(cv2.matchTemplate(stored, face.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0])
        match  = score >= 0.85
        return {"success": True, "verified": match, "score": round(score, 4),
                "message": "Face matched!" if match else "Face did not match."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Server error: {str(e)}"})


@app.delete('/delete/{name}')
async def delete(name: str):
    try:
        db = load_db()
        if name not in db:
            return JSONResponse(status_code=404, content={"success": False, "message": f"'{name}' not found."})

        del db[name]
        save_db(db)
        return {"success": True, "message": f"'{name}' deleted!"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Server error: {str(e)}"})
