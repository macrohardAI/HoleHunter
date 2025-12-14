import folium
import json
import os
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from .gps_utils import GPSHelper


class MapGenerator:
    HISTORY_FILE = 'peta_history.json'

    @staticmethod
    def encode_image(image_path, max_size=(300, 300)):
        """Ubah gambar jadi base64 string untuk HTML"""
        try:
            img = Image.open(image_path)
            img.thumbnail(max_size)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=70)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            return ""

    @staticmethod
    def load_history():
        if os.path.exists(MapGenerator.HISTORY_FILE):
            try:
                with open(MapGenerator.HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    @staticmethod
    def save_history(data):
        with open(MapGenerator.HISTORY_FILE, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def generate_map(new_predictions, output_file='peta_sebaran_lubang.html'):
        print(f"\n🗺️  Memproses data peta (Filter + Foto)...")

        # 1. Load History & Filter Duplikat
        history_points = MapGenerator.load_history()
        existing_files = {item['img_name'] for item in history_points}

        new_points = []
        for item in new_predictions:
            img_path = item.get('image_path')
            if not img_path: continue

            img_name = Path(img_path).name
            if img_name in existing_files: continue

            coords = GPSHelper.get_coordinates(img_path)
            if coords:
                lat, lon = coords
                new_points.append({
                    'lat': lat, 'lon': lon,
                    'class': item['class'],
                    'conf': float(item['confidence']),
                    'img_name': img_name,
                    'full_path': str(img_path)
                })

        total_points = history_points + new_points

        if new_points:
            MapGenerator.save_history(total_points)
            print(f"✅ Menambahkan {len(new_points)} titik baru.")

        if not total_points:
            print("⚠️ Tidak ada data GPS. Peta batal dibuat.")
            return

        # 2. Setup Peta
        avg_lat = sum(p['lat'] for p in total_points) / len(total_points)
        avg_lon = sum(p['lon'] for p in total_points) / len(total_points)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15)

        # 3. Setup Layers (Filter)
        layer_severe = folium.FeatureGroup(name='Severe (Parah)', show=True)
        layer_medium = folium.FeatureGroup(name='Medium (Sedang)', show=True)
        layer_normal = folium.FeatureGroup(name='Normal (Aman)', show=False)

        print("   ⏳ Menanamkan gambar ke peta...")

        for point in total_points:
            cls = point['class'].lower()

            # Tentukan Warna & Layer
            if cls == 'severe':
                color, icon, target = 'red', 'exclamation-sign', layer_severe
            elif cls == 'medium':
                color, icon, target = 'orange', 'warning-sign', layer_medium
            else:
                color, icon, target = 'green', 'ok-sign', layer_normal

            # Generate Gambar
            img_src = ""
            if 'full_path' in point and os.path.exists(point['full_path']):
                img_src = MapGenerator.encode_image(point['full_path'])

            # Popup HTML
            popup_html = f"""
            <div style="font-family: Arial; width: 300px;">
                <h4 style="margin:0;">Status: {point['class'].upper()}</h4>
                <p style="margin:5px 0;">Akurasi: <b>{point['conf']:.2%}</b></p>
                <div style="text-align:center; margin-top:10px; border:1px solid #ccc;">
                    <img src="{img_src}" style="width:100%; display:block;" alt="Foto tidak ditemukan">
                </div>
                <p style="font-size:10px; color:gray;">{point['img_name']}</p>
            </div>
            """

            iframe = folium.IFrame(popup_html, width=320, height=350)
            popup = folium.Popup(iframe, max_width=320)

            folium.Marker(
                location=[point['lat'], point['lon']],
                popup=popup,
                tooltip=point['class'].capitalize(),
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(target)

        # 4. Finalisasi
        layer_severe.add_to(m)
        layer_medium.add_to(m)
        layer_normal.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

        m.save(output_file)
        print(f"✅ Peta disimpan di: {output_file}")