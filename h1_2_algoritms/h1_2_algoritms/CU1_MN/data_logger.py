import csv
import os

class DataLogger:

    def __init__(self, filename="log_data.csv"):

        self.filename = filename
        self.data = []

        # Header (MUY importante para análisis después)
        self.header = [
            "time",
            "arm",
            "method",

            # Deseado
            "xd_x", "xd_y", "xd_z", "xd_qw", "xd_qx", "xd_qy", "xd_qz",

            # Real
            "x_x", "x_y", "x_z", "x_qw", "x_qx", "x_qy", "x_qz",

            # Error
            "e_x", "e_y", "e_z", "e_rx", "e_ry", "e_rz",

            # Norma error
            "error_norm"
        ]

    def log(self, t, arm_name, method, xd, x, e):

        row = [
            t,
            arm_name,
            method,

            *xd.tolist(),
            *x.tolist(),
            *e.tolist(),
            float((e**2).sum()**0.5)
        ]

        self.data.append(row)

    def save(self):

        os.makedirs("logs", exist_ok=True)
        filepath = os.path.join("logs", self.filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(self.data)

        print(f"Datos guardados en: {filepath}")