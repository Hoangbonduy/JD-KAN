import os

import numpy as np
import matplotlib.pyplot as plt


def _select_series(arr: np.ndarray) -> np.ndarray:
	if arr.ndim == 1:
		return arr
	if arr.ndim == 2:
		return arr[0]
	series = arr[0]
	if series.ndim > 1:
		return series[:, 0]
	return series


def main() -> None:
	base_dir = os.path.join(os.path.dirname(__file__), "..", "KQ")
	base_dir = os.path.abspath(base_dir)

	pred_ms_path = os.path.join(base_dir, "pred_MSJDKAN.npy")
	true_ms_path = os.path.join(base_dir, "true_MSJDKAN.npy")
	pred_time_path = os.path.join(base_dir, "pred_TimeKAN.npy")
	true_time_path = os.path.join(base_dir, "true_TimeKAN.npy")

	pred_ms = np.load(pred_ms_path)
	true_ms = np.load(true_ms_path)
	pred_time = np.load(pred_time_path)
	true_time = np.load(true_time_path)

	pred_ms_series = _select_series(pred_ms)
	true_ms_series = _select_series(true_ms)
	pred_time_series = _select_series(pred_time)
	true_time_series = _select_series(true_time)

	fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

	axes[0].plot(true_ms_series, label="True MSJDKAN", linewidth=2)
	axes[0].plot(pred_ms_series, label="Pred MSJDKAN", linewidth=1)
	axes[0].set_title("MSJDKAN")
	axes[0].legend()
	axes[0].grid(True, alpha=0.3)

	axes[1].plot(true_time_series, label="True TimeKAN", linewidth=2)
	axes[1].plot(pred_time_series, label="Pred TimeKAN", linewidth=1)
	axes[1].set_title("TimeKAN")
	axes[1].legend()
	axes[1].grid(True, alpha=0.3)

	fig.suptitle("Prediction vs Ground Truth", fontsize=14)
	fig.tight_layout(rect=[0, 0.02, 1, 0.95])

	output_path = os.path.join(base_dir, "prediction_plot.png")
	fig.savefig(output_path, dpi=200)
	plt.show()


if __name__ == "__main__":
	main()
