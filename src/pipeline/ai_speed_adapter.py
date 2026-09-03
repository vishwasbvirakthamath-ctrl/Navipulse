import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi

class AISpeedEstimator:
    """
    Online inference adapter for the trained Random Forest / ML model.
    Converts streaming IMU windows into the 40 causal features expected by idr_comprehensive_model.pkl.
    """
    def __init__(self, model_path: Path = None):
        if model_path is None:
            # Default to AI Model directory
            model_path = Path(__file__).resolve().parent.parent.parent / "AI Model" / "idr_comprehensive_model.pkl"
        
        self.model_path = Path(model_path)
        self.is_loaded = False
        self.model = None
        self.features = []
        self.targets = []
        self.sample_rate_hz = 10.0
        
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                artifact = pickle.load(f)
            self.model = artifact["model"]
            self.model.n_jobs = 1
            self.features = artifact["features"]
            self.targets = artifact["targets"]
            self.sample_rate_hz = artifact.get("sample_rate_hz", 10.0)
            self.is_loaded = True
            print(f"[AISpeedEstimator] Loaded model from {self.model_path} with {len(self.features)} features.")
        else:
            print(f"[AISpeedEstimator] WARNING: Model not found at {self.model_path}")

    def extract_features(self, window: np.ndarray, yaw_rad: float = 0.0) -> np.ndarray:
        """
        Extract the 40 causal features from an IMU window (N, 6) [ax, ay, az, gx, gy, gz].
        """
        # window shape (N, 6)
        N = len(window)
        ax = window[:, 0]
        ay = window[:, 1]
        az = window[:, 2]
        gx = window[:, 3]
        gy = window[:, 4]
        gz = window[:, 5]
        
        accel_mag = np.linalg.norm(window[:, :3], axis=1)
        gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)
        
        feature_dict = {}
        sources = {
            "accel_x": ax, "accel_y": ay, "accel_z": az,
            "gyro_x": gx, "gyro_y": gy, "gyro_z": gz,
            "accel_mag": accel_mag, "gyro_mag": gyro_mag
        }
        
        fs = self.sample_rate_hz
        cutoff_hz = 2.5
        sos = butter(4, cutoff_hz, fs=fs, btype="low", output="sos")
        
        for name, data in sources.items():
            # Causal lowpass filter initialized at first sample
            filtered = sosfilt(sos, data, zi=sosfilt_zi(sos) * data[0])[0]
            feature_dict[f"{name}_filtered"] = filtered[-1] # current step
            # Rolling mean/std over the window
            feature_dict[f"{name}_mean"] = np.mean(filtered)
            feature_dict[f"{name}_std"] = np.std(filtered)
            feature_dict[f"{name}_delta"] = filtered[-1] - filtered[-2] if N > 1 else 0.0
            
        # Orientation features
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        feature_dict["cos_yaw_filtered"] = cos_y
        feature_dict["sin_yaw_filtered"] = sin_y
        feature_dict["cos_yaw_mean"] = cos_y
        feature_dict["sin_yaw_mean"] = sin_y
        feature_dict["cos_yaw_std"] = 0.0
        feature_dict["sin_yaw_std"] = 0.0
        feature_dict["cos_yaw_delta"] = 0.0
        feature_dict["sin_yaw_delta"] = 0.0
        
        # Assemble DataFrame with column names in exact order expected by the model
        df = pd.DataFrame([feature_dict])[self.features]
        return df

    def predict_speed(self, window: np.ndarray, yaw_rad: float = 0.0) -> float:
        """
        Predict forward speed in metres/second.
        """
        if not self.is_loaded or len(window) < 5:
            return None
            
        feat_df = self.extract_features(window, yaw_rad)
        preds = self.model.predict(feat_df) # shape (1, 3)
        # Target 0 is speed in km/h
        speed_kmh = float(preds[0, 0])
        speed_mps = max(0.0, speed_kmh / 3.6)
        return speed_mps

    def __call__(self, window: np.ndarray, yaw_rad: float = 0.0) -> float:
        return self.predict_speed(window, yaw_rad)

if __name__ == "__main__":
    estimator = AISpeedEstimator(Path("AI Model/idr_comprehensive_model.pkl"))
    # Dummy window (50, 6)
    dummy_window = np.zeros((50, 6))
    dummy_window[:, 2] = 9.81
    speed = estimator.predict_speed(dummy_window, yaw_rad=0.0)
    print(f"Predicted speed on stationary window: {speed:.3f} m/s ({speed*3.6:.2f} km/h)")
