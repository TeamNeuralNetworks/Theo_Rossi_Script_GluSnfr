import os, sys, glob, zipfile
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path when running from a different CWD
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from Feature_extraction.extract_metrics import extract_metrics


# -------------------------
# Helpers
# -------------------------

def _is_valid_xlsx(path: str) -> bool:
	try:
		with zipfile.ZipFile(path) as z:
			return '[Content_Types].xml' in z.namelist()
	except Exception:
		return False


def _load_time_trials_from_xlsx(xlsx_path: str):
	df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
	if df.shape[1] < 2:
		raise ValueError(f"Expected 2 columns (data + time) in {xlsx_path}")
	t_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').to_numpy(float)
	X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').to_numpy(float)
	ok = np.isfinite(t_raw)
	time = t_raw[ok]
	trials = X[ok, :]
	return time, trials


# -------------------------
# Config (generic paths)
# -------------------------

SINGLE_FILE = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_4Ca\20211125_linescan1_20Hz_10pulses_4mMCa_bouton1_traces_converted.xlsx"
IN_DIR = r"C:\Users\Antoine.Valera\Desktop\PPR_DATA_FINAL\Theo_4Ca\\"
OUT_DIR = r"C:\Users\Antoine.Valera\Desktop\Testout"

TRAIN_START_S = 0.5
ISI_S = 0.05
N_PULSES = 10


# -------------------------
# Example 1: Single-file basic refit and overlay
# -------------------------

def example_basic_single_file(xlsx_path: str, out_dir: str):
	if not _is_valid_xlsx(xlsx_path):
		print(f"[skip] Not a valid .xlsx package: {xlsx_path}")
		return
	time, trials = _load_time_trials_from_xlsx(xlsx_path)

	res = extract_metrics(
		time, trials,
		train_start=TRAIN_START_S, isi=ISI_S, n_pulses=N_PULSES,
		options={'normalize_dff': True, 'bleach': True, 'plot': {'enabled': False}}
	)

	t = res['time_s']
	y_avg = res['average']['y_avg']
	yhat = res['average']['yhat_avg']

	os.makedirs(out_dir, exist_ok=True)
	base = os.path.splitext(os.path.basename(xlsx_path))[0]
	fig, ax = plt.subplots(figsize=(8, 4))
	ax.plot(t, y_avg, label='average', color='k', alpha=0.7)
	ax.plot(t, yhat, label='model (default)', color='tab:blue')
	ax.set_title(f"{base} | default fit")
	ax.set_xlabel('Time (s)'); ax.set_ylabel('9F/F0')
	ax.legend(loc='best')
	out_png = os.path.join(out_dir, f"{base}_default_refit.png")
	fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
	print(f"[ok] Saved {out_png}")


# -------------------------
# Example 2: Refit with alternative kinetics grids and compare
# -------------------------

def example_compare_kinetics(xlsx_path: str, out_dir: str):
	if not _is_valid_xlsx(xlsx_path):
		print(f"[skip] Not a valid .xlsx package: {xlsx_path}")
		return
	time, trials = _load_time_trials_from_xlsx(xlsx_path)

	# Default fit
	res_def = extract_metrics(
		time, trials,
		train_start=TRAIN_START_S, isi=ISI_S, n_pulses=N_PULSES,
		options={'normalize_dff': True, 'bleach': True, 'plot': {'enabled': False}}
	)

	# Alternative kinetics: extend tau_d grid and slope grid
	res_alt = extract_metrics(
		time, trials,
		train_start=TRAIN_START_S, isi=ISI_S, n_pulses=N_PULSES,
		options={
			'normalize_dff': True,
			'bleach': True,
			'kin_taur_grid_ms': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
			'kin_taud0_grid_ms': [1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
			'kin_slope_grid_ms': [0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
			'plot': {'enabled': False},
		}
	)

	t = res_def['time_s']
	y_avg = res_def['average']['y_avg']
	yhat_def = res_def['average']['yhat_avg']
	yhat_alt = res_alt['average']['yhat_avg']

	os.makedirs(out_dir, exist_ok=True)
	base = os.path.splitext(os.path.basename(xlsx_path))[0]
	fig, ax = plt.subplots(figsize=(8, 4))
	ax.plot(t, y_avg, label='average', color='k', alpha=0.65)
	ax.plot(t, yhat_def, label='model (default)', color='tab:blue')
	ax.plot(t, yhat_alt, label='model (alt kinetics)', color='tab:orange')
	ax.set_title(f"{base} | default vs alt kinetics")
	ax.set_xlabel('Time (s)'); ax.set_ylabel('9F/F0')
	ax.legend(loc='best')
	out_png = os.path.join(out_dir, f"{base}_compare_kinetics.png")
	fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
	print(f"[ok] Saved {out_png}")


# -------------------------
# Example 3: Simple folder batch (default refit per file)
# -------------------------

def example_folder_batch(in_dir: str, out_dir: str):
	os.makedirs(out_dir, exist_ok=True)
	for xlsx_path in glob.glob(os.path.join(in_dir, "*.xlsx")):
		try:
			example_basic_single_file(xlsx_path, out_dir)
		except Exception as e:
			print(f"[warn] Skipping {xlsx_path}: {e}")


if __name__ == "__main__":
	# Adjust SINGLE_FILE / IN_DIR / OUT_DIR above, then uncomment what you want to run.
	# example_basic_single_file(SINGLE_FILE, OUT_DIR)
	# example_compare_kinetics(SINGLE_FILE, OUT_DIR)
	# example_folder_batch(IN_DIR, OUT_DIR)
	pass
