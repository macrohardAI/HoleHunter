from PIL import Image, ExifTags


class GPSHelper:
    @staticmethod
    def get_decimal_from_dms(dms, ref):
        """Convert Degree-Minute-Second to Decimal"""
        degrees = dms[0]
        minutes = dms[1] / 60.0
        seconds = dms[2] / 3600.0

        decimal = degrees + minutes + seconds
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal

    @staticmethod
    def get_coordinates(image_path):
        """Extract Latitude and Longitude from Image"""
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()

            if not exif_data:
                return None

            
            gps_info = {}
            for tag, value in exif_data.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_info = value
                    break

            if not gps_info:
                return None

            lat_ref = gps_info.get(1)
            lat_dms = gps_info.get(2)
            lon_ref = gps_info.get(3)
            lon_dms = gps_info.get(4)

            if lat_ref and lat_dms and lon_ref and lon_dms:
                lat = GPSHelper.get_decimal_from_dms(lat_dms, lat_ref)
                lon = GPSHelper.get_decimal_from_dms(lon_dms, lon_ref)
                return lat, lon

            return None

        except Exception:
            return None