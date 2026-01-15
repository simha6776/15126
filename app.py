# app.py
import os
import time
import threading
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2

from detector.yolo_detector import YOLODetector
from detector.tracker import TrackerWrapper
from detector.ocr import read_plate
from database import Session, events, shipments, users, stops, inventory,maintenance

from modules.route_optimizer import nearest_neighbor,two_opt,tour_length
from modules.predictive_maintenance import predict as predict_maint

from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

import pandas as pd
from modules.predictive_maintenance import train_model




LAST_RESULT = {
    "status": "idle",
    "plate": None
}


df = pd.read_csv('data/training_data.csv')
model = train_model(df)
print("✅ Model trained and saved successfully as models/maint_model.pkl")



app = Flask(__name__)
app.secret_key = "super_secret_key_123" 
CORS(app)

vehicle_model_path="yolov8n.pt"
plate_model_path="license_plate_detector.pt"
DETECT_CONF = 0.4

CAMERA_LOCATIONS = {
    "cam1": "Bengaluru Hub A",
    "cam2": "Mysuru Warehouse",
    "cam3": "Chennai Sorting Center",
    "cam4": "Mumbai Distribution Point",
}

detector = YOLODetector(vehicle_model_path="yolov8n.pt", plate_model_path="license_plate_detector.pt")
tracker = TrackerWrapper()
processing = {}

def ensure_upload_folder():
    os.makedirs('uploads', exist_ok=True)

def process_video(filepath, camera_id="cam1"):
    sess = Session()
    cap = cv2.VideoCapture(filepath)
    frame_no = 0
    last_plate = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1

            vehicles = detector.detect_vehicles(frame)

            for (vx1,vy1,vx2,vy2) in vehicles:
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                if vehicle_crop.size == 0:
                    continue

                plates = detector.detect_plates(vehicle_crop)

                for (px1,py1,px2,py2) in plates:
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    plate_text = read_plate(plate_crop)

                    if not plate_text or len(plate_text)<6:
                      continue

                    last_plate = plate_text.strip()

                    location_name = CAMERA_LOCATIONS.get(camera_id, "Unknown Location")

                    sess.execute(
                        events.insert().values(
                            object_id="vehicle",
                            object_type="vehicle",
                            plate=last_plate,
                            camera_id=camera_id,
                            location=location_name,
                            timestamp=datetime.utcnow(),
                            frame=frame_no,
                            x=float(vx1), y=float(vy1),
                            w=float(vx2-vx1), h=float(vy2-vy1)
                        )
                    )

            sess.commit()

    finally:
        cap.release()
        sess.close()
        processing.pop(filepath, None)


    # ✅ SAVE LAST PLATE RESULT
    if last_plate:
        processing["last_result"] = f"The vehicle number plate is : {last_plate}"
    if last_plate:
        LAST_RESULT["status"] = "completed"
        LAST_RESULT["plate"] = last_plate
    else:
        LAST_RESULT["status"] = "completed"
        LAST_RESULT["plate"] = "Not detected"

#@app.route('/')
#def index():
    #"""Main dashboard page"""
    #return render_template('index.html')

#@app.route('/')
#def  dashboard():
    #if 'user' not in session:
       # return redirect('/login')
    #return render_template('index.html')

@app.route('/')
def root():
    return redirect('/login')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')




@app.route('/supply_chain.html')
def supply_chain_page():
    return render_template('supply_chain.html')

@app.route('/inventory.html')
def inventory_page():
    return render_template('inventory.html')

@app.route('/maintenance.html')
def maintenance_page():
    return render_template('maintenance.html')

@app.route('/route_optimizer')
def route_optimizer_page():
    return render_template('route_optimizer.html')


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    cam = request.form.get('camera_id', 'cam1')
    if not f:
        return jsonify({"error":"no file"}), 400
    ensure_upload_folder()
    path = os.path.join('uploads', f.filename)
    f.save(path)
    t = threading.Thread(target=process_video, args=(path, cam), daemon=True)
    t.start()
    processing[path] = {"thread": t, "camera": cam, "started": time.time()}
    return jsonify({
    "status": "processing",
    "message": "Video uploaded successfully. Processing started."
})
@app.route('/result', methods=['GET'])
def get_result():
    if LAST_RESULT["status"] != "completed":
        return jsonify({"status": "processing"})
    return jsonify({
        "status": "completed",
        "message": f"The vehicle number plate is : {LAST_RESULT['plate']}"
    })


@app.route('/events/search', methods=['GET'])
def search_events():
    q_plate = request.args.get('plate')
    sess = Session()
    query = events.select()
    if q_plate:
        query = query.where(events.c.plate.like(f"%{q_plate}%"))
    res = sess.execute(query).fetchall()
    out = []
    for r in res:
        out.append({
            "id": r.id,
            "object_id": r.object_id,
            "object_type": r.object_type,
            "plate": r.plate,
            "camera_id": r.camera_id,
            "location": r.location,
            "timestamp": str(r.timestamp),
            "frame": r.frame,
            "bbox": {"x": r.x, "y": r.y, "w": r.w, "h": r.h}
        })
    sess.close()
    return jsonify(out)

@app.route('/location', methods=['GET'])
def get_location():
    plate = request.args.get('plate')
    if not plate:
        return jsonify({"error": "Plate number required"}), 400
    sess = Session()
    query = events.select().where(events.c.plate.like(f"%{plate}%")).order_by(events.c.timestamp.desc()).limit(1)
    result = sess.execute(query).fetchone()
    sess.close()
    if result:
        return jsonify({
            "plate": result.plate,
            "last_seen": str(result.timestamp),
            "location": result.location,
            "camera_id": result.camera_id,
            "object_type": result.object_type
        })
    else:
        return jsonify({"message": "No record found for this plate"})
    
@app.route('/events/clear', methods=['POST'])
def clear_events():
     sess = Session()
     sess.execute(events.delete())
     sess.commit()
     sess.close()
     return jsonify({"status":"all events deleted"})
    
    # shipments CRUD
# 🚚 SHIPMENTS ENDPOINTS (working version)


from sqlalchemy import select

@app.route('/shipments', methods=['GET'])
def get_shipments():
    """List all shipments"""
    sess = Session()
    rows = sess.execute(select(shipments)).fetchall()
    sess.close()
    return jsonify([dict(r._mapping) for r in rows])


@app.route('/shipments', methods=['POST'])
def create_shipment():
    """Create a new shipment"""
    data = request.get_json() or {}
    sess = Session()
    ins = shipments.insert().values(
        shipment_code=data.get('code'),
        origin=data.get('origin'),
        destination=data.get('destination'),
        status='created',
        assigned_plate=data.get('assigned_plate')
    )
    sess.execute(ins)
    sess.commit()
    sess.close()
    return jsonify({'message': 'Shipment created successfully!'})


@app.route('/shipments/<int:sid>', methods=['GET'])
def get_shipment(sid):
    """Get single shipment by ID"""
    sess = Session()
    row = sess.execute(select(shipments).where(shipments.c.id == sid)).fetchone()
    sess.close()
    if not row:
        return jsonify({'error': 'Shipment not found'}), 404
    return jsonify(dict(row._mapping))


# predictive maintenance endpoint
@app.route('/predict_maintenance', methods=['POST'])
def predict_maintenance():
    data = request.get_json() or {}
    return jsonify(predict_maint(data))

# inventory endpoints
#@app.route('/inventory', methods=['GET', 'POST'])
#def inventory_list_create():
    #sess = Session()
    #if request.method == 'GET':
        #rows = sess.execute(inventory.select()).fetchall()
        #sess.close()
       # out = [dict(r) for r in rows]
        #return jsonify(out)
   # data = request.get_json() or {}
   #sess.execute(inventory.insert().values(sku=data.get('sku'), name=data.get('name'), qty=int(data.get('qty',0)), location=data.get('location',''))); sess.commit(); sess.close()
    #return jsonify({'status':'ok'})

#from modules import route_optimizer

#@app.route('/route_optimizer.html')
#    return render_template('route_optimizer.html')

#@app.route('/optimize_route', methods=['POST'])
#def optimize_route():
    #data = request.get_json()
    #points = data.get('points', [])
    #if not points or len(points) < 2:
        #return jsonify({'error': 'Need at least two points to optimize'}), 400

    #try:
       # tour = route_optimizer.nearest_neighbor(points)
        #improved = route_optimizer.two_opt(points, tour)
        #total_dist = route_optimizer.tour_length(points, improved)
       # return jsonify({'tour': improved, 'total_distance': round(total_dist, 2)})
    #except Exception as e:
        #return jsonify({'error': str(e)}), 500

from geopy.geocoders import Nominatim
from modules import route_optimizer
from flask import request, jsonify
import math

@app.route("/optimize_route", methods=["POST"])
def optimize_route():
    data = request.get_json()
    locations = data.get("locations")

    if not locations or len(locations) < 2:
        return jsonify({"error": "Need at least two locations"}), 400

    geolocator = Nominatim(user_agent="logistics-ai-tracking")
    points = []
    geo_info = []

    for loc in locations:
        geo = geolocator.geocode(loc)
        if geo:
            points.append([geo.latitude, geo.longitude])
            geo_info.append({
                "name": loc,
                "lat": geo.latitude,
                "lon": geo.longitude
            })
        else:
            return jsonify({"error": f"Could not locate {loc}"}), 400

    # Run optimization
    tour = route_optimizer.nearest_neighbor(points)
    optimized_points = [geo_info[i] for i in tour]
    total_distance = route_optimizer.tour_length(points, tour)

    return jsonify({
        "optimized_order": optimized_points,
        "total_distance_km": round(total_distance, 2)
    })

# app.py
#from flask import Flask, render_template, request, jsonify
#from modules.inventory_manager import InventoryManager
#from flask_cors import CORS

#inv = InventoryManager()

#@app.route("/")
#def home():
   # return render_template("inventory.html")

#@app.route("/inventory", methods=["GET"])
#def get_inventory():
    #return jsonify(inv.get_all_items())

#@app.route("/add", methods=["POST"])
#def add():
    #data = request.json
    #inv.add_item(data["name"], data["quantity"], data["price"], data["category"])
    #return jsonify({"message": "Item added successfully"})

#@app.route("/update/<int:item_id>", methods=["POST"])
#def update(item_id):
    #data = request.json
    #inv.update_item(item_id, data["name"], data["quantity"], data["price"], data["category"])
    #return jsonify({"message": "Item updated successfully"})

#@app.route("/delete/<int:item_id>", methods=["DELETE"])
#def delete_item(item_id):
    #inv.delete_item(item_id)
    #return jsonify({"message": "Item deleted successfully"})

#@app.route("/search", methods=["GET"])
#def search_item():
    #keyword = request.args.get("q", "")
   # return jsonify(inv.search_item(keyword))

#@app.route("/reset", methods=["POST"])
#def reset():
   # inv.reset_inventory()
    #return jsonify({"message": "Inventory reset successfully"})

#@app.route("/sale/<int:item_id>", methods=["POST"])
#def sale(item_id):
   # qty = request.json.get("qty", 0)
   # ok = inv.record_sale(item_id, qty)
    #if ok:
     #   return jsonify({"message": "Sale recorded"})
#    return jsonify({"error": "Not enough stock"}), 400

#@app.route("/alerts", methods=["GET"])
#def low_stock():
    #return jsonify(inv.low_stock_alerts())

from modules.inventory_manager import InventoryManager


inv = InventoryManager()

#@app.route("/")
#def home():
    #return render_template("inventory.html")

@app.route("/inventory", methods=["GET"])
def get_inventory():
    return jsonify(inv.get_all_items())

@app.route("/add", methods=["POST"])
def add_item():
    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "Invalid input"}), 400
    inv.add_item(data["name"], data["quantity"], data["price"], data["category"])
    return jsonify({"message": "Item added successfully"})

@app.route("/update/<int:item_id>", methods=["POST"])
def update_item(item_id):
    data = request.get_json()
    inv.update_item(item_id, data["name"], data["quantity"], data["price"], data["category"])
    return jsonify({"message": "Item updated successfully"})

@app.route("/delete/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):
    inv.delete_item(item_id)
    return jsonify({"message": "Item deleted successfully"})

@app.route("/search", methods=["GET"])
def search_inventory():
    keyword = request.args.get("q", "")
    return jsonify(inv.search_item(keyword))

@app.route("/reset", methods=["POST"])
def reset_inventory():
    inv.reset_inventory()
    return jsonify({"message": "Inventory reset successfully"})

@app.route("/sale/<int:item_id>", methods=["POST"])
def sale(item_id):
    qty = request.json.get("qty", 0)
    ok = inv.record_sale(item_id, qty)
    if ok:
        return jsonify({"message": "Sale recorded"})
    return jsonify({"error": "Not enough stock"}), 400

@app.route("/alerts", methods=["GET"])
def low_stock_alerts():
    return jsonify(inv.low_stock_alerts())

from werkzeug.security import generate_password_hash, check_password_hash
from flask import session, redirect, url_for, flash

app.secret_key = "my_super_secret_key"

@app.before_request
def require_login():
    if request.path.startswith('/static'):
        return

    public_routes = ('/login', '/register', '/forgot')
    if not any(request.path.startswith(r) for r in public_routes):
        if 'user' not in session:
            return redirect('/login')




#@app.route('/')
#def index():
    #return render_template('index.html')


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        sess = Session()
        user = sess.execute(
            users.select().where(users.c.username == username)
        ).fetchone()
        sess.close()

        if user and check_password_hash(user.password, password):
            session['user'] = username
            return redirect('/dashboard')
        else:
            flash("Invalid credentials", "error")

    return render_template('login.html')



@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        sess = Session()
        try:
            # 🔎 check existing user
            existing = sess.execute(
                users.select().where(
                    (users.c.username == username) |
                    (users.c.email == email)
                )
            ).fetchone()

            if existing:
                flash("Username or Email already exists", "error")
                return render_template('register.html')

            sess.execute(users.insert().values(
                username=username,
                email=email,
                password=password
            ))
            sess.commit()
            flash("Registration successful. Please login.", "success")
            return redirect('/login')

        except Exception as e:
            sess.rollback()
            flash("Registration failed", "error")

        finally:
            sess.close()

    return render_template('register.html')



@app.route('/forgot', methods=['GET','POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')

        sess = Session()
        user = sess.execute(
            users.select().where(users.c.email == email)
        ).fetchone()
        sess.close()

        if user:
            flash("Password reset link sent (demo)", "success")
        else:
            flash("Email not found", "error")

    return render_template('forgot_password.html')



@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
