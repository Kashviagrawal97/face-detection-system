import os, json, cv2, numpy as np, uuid
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from face_detection import extract_face

app     = FastAPI()
DB_PATH = os.path.join(cv2.data.haarcascades, 'users.json')

def load_db(): return json.load(open(DB_PATH)) if os.path.exists(DB_PATH) else {}
def save_db(d): json.dump(d, open(DB_PATH, 'w'), indent=2)

def get_user_by_id(user_id, db):
    user_id = user_id.strip().strip('"').strip("'")   
    for name, data in db.items():
        if isinstance(data, dict) and data.get('id') == user_id:
            return name, data
    return None, None

@app.post('/register')
async def register(name: str = Form(...), image: UploadFile = File(...)):
    try:
        name = name.strip().lower()
        db   = load_db()

        if name in db:
            return JSONResponse(status_code=409, content={"success": False, "message": f"'{name}' already registered."})

        face, err = extract_face(await image.read())
        if err: return JSONResponse(status_code=422, content={"success": False, "message": err})

        user_id  = str(uuid.uuid4())
        db[name] = {"id": user_id, "face": face.flatten().tolist()}
        save_db(db)
        return {
            "success": True,
            "message": f"'{name}' registered successfully!",
            "your_id": user_id,
            "note": f"Copy your ID to delete account: {user_id}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Server error: {str(e)}"})


@app.post('/verify')
async def verify(image: UploadFile = File(...)):
    try:
        db = load_db()
        if not db:
            return JSONResponse(status_code=404, content={"success": False, "message": "No users registered yet."})

        face, err = extract_face(await image.read())
        if err: return JSONResponse(status_code=422, content={"success": False, "message": err})

        best_name, best_id, best_score = None, None, 0.0

        for name, data in db.items():
           
            if isinstance(data, dict):
                face_list = data['face']
                uid       = data['id']
            else:
                face_list = data       
                uid       = None

            stored = np.array(face_list, dtype=np.float32).reshape(100, 100)
            score  = float(cv2.matchTemplate(stored, face.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0])

            if score > best_score:
                best_score = score
                best_name  = name
                best_id    = uid

        match = best_score >= 0.85
        return {
            "success":  True,
            "verified": match,
            "name":     best_name if match else None,
            "id":       best_id   if match else None,
            "score":    round(best_score, 4),
            "message":  f"Face matched with '{best_name}'!" if match else "No match found."
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Server error: {str(e)}"})


@app.delete('/delete')
async def delete(user_id: str):
    try:
        db      = load_db()
        name, _ = get_user_by_id(user_id, db)   
        if not name:
            return JSONResponse(status_code=404, content={"success": False, "message": f"ID '{user_id}' not found."})

        del db[name]
        save_db(db)
        return {"success": True, "message": f"User '{name}' deleted successfully!"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Server error: {str(e)}"})