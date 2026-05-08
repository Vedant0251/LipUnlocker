import numpy as np
import glob
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

WINDOW_SIZE = 60 
FEATURES = 40
EPOCHS = 150

def get_sliced_dataset():
    X, y = [], []
    
    for f in glob.glob('data/password/*.npy'):
        try:
            data = np.load(f)
            if data.shape[-1] != FEATURES:
                print(f"  Skipping {f}: wrong feature dimension {data.shape} — delete old data and re-record!")
                continue
            if len(data) < WINDOW_SIZE:
                padded = np.zeros((WINDOW_SIZE, FEATURES))
                padded[:len(data), :] = data
                X.append(padded); y.append(1)
            else:
                step = WINDOW_SIZE // 2
                for i in range(0, len(data) - WINDOW_SIZE + 1, step):
                    X.append(data[i : i + WINDOW_SIZE]); y.append(1)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    for f in glob.glob('data/random/*.npy'):
        try:
            data = np.load(f)
            if data.shape[-1] != FEATURES:
                print(f"  Skipping {f}: wrong feature dimension {data.shape} — delete old data and re-record!")
                continue
            if len(data) < WINDOW_SIZE:
                padded = np.zeros((WINDOW_SIZE, FEATURES))
                padded[:len(data), :] = data
                X.append(padded); y.append(0)
            else:
                step = WINDOW_SIZE // 2
                for i in range(0, len(data) - WINDOW_SIZE + 1, step):
                    X.append(data[i : i + WINDOW_SIZE]); y.append(0)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return np.array(X), np.array(y)

print("Loading dataset...")
X_train, y_train = get_sliced_dataset()

if len(X_train) == 0:
    print("\nNo valid data found!")
    print("Please DELETE your data/password and data/random folders and re-record using record_data.py!")
    exit(1)

n_pass = int(np.sum(y_train == 1))
n_rand = int(np.sum(y_train == 0))
print(f"Generated {len(X_train)} sequences | Password: {n_pass} | Random: {n_rand}")

if n_pass == 0 or n_rand == 0:
    print("ERROR: Need BOTH password and random samples to train!")
    exit(1)

# Class weights to handle any imbalance
total = n_pass + n_rand
class_weights = {0: total / (2.0 * n_rand), 1: total / (2.0 * n_pass)}
print(f"Class weights applied: {class_weights}")

print("\nBuilding Simple LSTM (appropriate for small dataset)...")
model = models.Sequential([
    layers.Input(shape=(WINDOW_SIZE, FEATURES)),
    layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.LSTM(64, return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                recurrent_regularizer=tf.keras.regularizers.l2(0.005)),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

print("\nStarting Deep Training ▶▶▶")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

best_val_acc = max(history.history['val_accuracy']) * 100
print(f"\nBest Validation Accuracy: {best_val_acc:.1f}%")

model.save('lip_model.h5')
model.save('lip_model.keras')
print("✔ Model saved as 'lip_model.h5' and 'lip_model.keras'")

# ─────────────────────────────────────────────
# COMPREHENSIVE MODEL EVALUATION
# ─────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

print("\n" + "=" * 55)
print("  COMPREHENSIVE MODEL EVALUATION REPORT")
print("=" * 55)

# Predictions on ALL data (train+val combined)
y_proba = model.predict(X_train, verbose=0).flatten()
y_pred  = (y_proba >= 0.5).astype(int)

acc       = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred, zero_division=0)
recall    = recall_score(y_train, y_pred, zero_division=0)
f1        = f1_score(y_train, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_train, y_proba)
avg_prec  = average_precision_score(y_train, y_proba)
cm        = confusion_matrix(y_train, y_pred)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
far         = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Accept Rate
frr         = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # False Reject Rate

print(f"  Accuracy         : {acc*100:.2f}%")
print(f"  Precision        : {precision*100:.2f}%")
print(f"  Recall (Sensitivity): {recall*100:.2f}%")
print(f"  Specificity      : {specificity*100:.2f}%")
print(f"  F1 Score         : {f1*100:.2f}%")
print(f"  ROC-AUC Score    : {roc_auc:.4f}")
print(f"  Avg Precision    : {avg_prec:.4f}")
print(f"  --- Security Metrics ---")
print(f"  False Accept Rate (FAR)  : {far*100:.2f}%  ← lower is more secure")
print(f"  False Reject Rate (FRR)  : {frr*100:.2f}%  ← lower is more usable")
print(f"\n{classification_report(y_train, y_pred, target_names=['Random','Password'])}")
print("=" * 55)

# ─────────────────────────────────────────────
# 6-PANEL VISUALIZATION DASHBOARD
# ─────────────────────────────────────────────
BG   = '#1e1e1e'
CARD = '#2a2a2a'
GREEN  = '#00e676'
GREEN2 = '#69f0ae'
RED    = '#ff1744'
RED2   = '#ff8a80'
BLUE   = '#2196f3'
AMBER  = '#ffc107'
WHITE  = '#ffffff'
GRAY   = '#9e9e9e'

def style_ax(ax, title=''):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    if title:
        ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.15, color=GRAY)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

fig = plt.figure(figsize=(18, 11), facecolor=BG)
fig.suptitle('DeepLipUnlocker — Model Evaluation Dashboard', color=WHITE, fontsize=16, fontweight='bold', y=0.98)

gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3)

# 1. Accuracy curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['accuracy'],     color=GREEN,  label='Train',      linewidth=2)
ax1.plot(history.history['val_accuracy'], color=GREEN2, label='Validation', linewidth=2, linestyle='--')
ax1.axhline(y=acc, color=AMBER, linestyle=':', linewidth=1.2, label=f'Final {acc*100:.1f}%')
style_ax(ax1, f'Accuracy  (Best Val: {best_val_acc:.1f}%)')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=8, facecolor='#333', labelcolor=WHITE)

# 2. Loss curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history.history['loss'],     color=RED,  label='Train',      linewidth=2)
ax2.plot(history.history['val_loss'], color=RED2, label='Validation', linewidth=2, linestyle='--')
style_ax(ax2, 'Loss Curve')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(fontsize=8, facecolor='#333', labelcolor=WHITE)

# 3. Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
im = ax3.imshow(cm, cmap='Blues', aspect='auto')
ax3.set_facecolor(CARD)
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color=WHITE, fontsize=18, fontweight='bold')
ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Pred: Random', 'Pred: Password'], color=WHITE)
ax3.set_yticklabels(['True: Random', 'True: Password'], color=WHITE)
ax3.set_title('Confusion Matrix', color=WHITE, fontsize=11, fontweight='bold')
for spine in ax3.spines.values(): spine.set_edgecolor('#444')

# 4. ROC Curve
ax4 = fig.add_subplot(gs[1, 0])
fpr_arr, tpr_arr, _ = roc_curve(y_train, y_proba)
ax4.plot(fpr_arr, tpr_arr, color=BLUE, linewidth=2, label=f'AUC = {roc_auc:.4f}')
ax4.plot([0, 1], [0, 1], color=GRAY, linestyle='--', linewidth=1)
ax4.fill_between(fpr_arr, tpr_arr, alpha=0.15, color=BLUE)
style_ax(ax4, 'ROC Curve')
ax4.set_xlabel('False Accept Rate (FAR)'); ax4.set_ylabel('True Accept Rate (TAR)')
ax4.legend(fontsize=9, facecolor='#333', labelcolor=WHITE)
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1.05)

# 5. Precision-Recall Curve
ax5 = fig.add_subplot(gs[1, 1])
prec_arr, rec_arr, _ = precision_recall_curve(y_train, y_proba)
ax5.plot(rec_arr, prec_arr, color=GREEN, linewidth=2, label=f'AP = {avg_prec:.4f}')
ax5.fill_between(rec_arr, prec_arr, alpha=0.15, color=GREEN)
style_ax(ax5, 'Precision-Recall Curve')
ax5.set_xlabel('Recall'); ax5.set_ylabel('Precision')
ax5.legend(fontsize=9, facecolor='#333', labelcolor=WHITE)
ax5.set_xlim(0, 1); ax5.set_ylim(0, 1.05)

# 6. Security KPI Bar Chart
ax6 = fig.add_subplot(gs[1, 2])
kpi_labels = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'ROC-AUC']
kpi_values = [acc, precision, recall, specificity, f1, roc_auc]
colors_kpi = [GREEN if v >= 0.85 else AMBER if v >= 0.70 else RED for v in kpi_values]
bars = ax6.barh(kpi_labels, kpi_values, color=colors_kpi, edgecolor='#444', height=0.55)
for bar, val in zip(bars, kpi_values):
    ax6.text(min(val + 0.01, 0.99), bar.get_y() + bar.get_height() / 2,
             f'{val*100:.1f}%', va='center', color=WHITE, fontsize=9, fontweight='bold')
ax6.set_xlim(0, 1.08)
ax6.axvline(x=0.85, color=AMBER, linestyle='--', linewidth=1, alpha=0.6)
style_ax(ax6, 'Security KPIs')
ax6.set_xlabel('Score')

plt.savefig('training_visualization.png', facecolor=BG, dpi=120, bbox_inches='tight')
print("\nVisualization dashboard saved to 'training_visualization.png'")
plt.show()

