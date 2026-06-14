import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': '#0B1410','axes.facecolor':'#121F18',
    'axes.edgecolor':'#1F3328','axes.labelcolor':'#D4F0DD',
    'xtick.color':'#6FA384','ytick.color':'#6FA384',
    'grid.color':'#1F3328','grid.alpha':0.6,'text.color':'#E8F5ED',
    'font.family':'DejaVu Sans','axes.titlepad':14,
    'axes.titlesize':12,'axes.titleweight':'bold',
})
GREEN,BLUE,ORANGE,RED,YELLOW = '#4ADE80','#38BDF8','#FB923C','#F87171','#FACC15'

np.random.seed(42)
timestamps = pd.date_range('2024-06-01', periods=7*48, freq='30min')
n = len(timestamps)

soil_moisture = np.zeros(n)
soil_moisture[0] = 55
THRESH = {'soil_moisture_low':30,'temp_high':35,'humidity_low':35,'water_level_low':15}

hours = (timestamps.hour + timestamps.minute/60).to_numpy()
temperature = 26 + 6*np.sin(2*np.pi*(hours-8)/24) + np.random.normal(0,0.8,n)
# Inject a heatwave on day 3 (indices 96-120) — pushes temp above 35
heatwave_mask = (np.arange(n) >= 96) & (np.arange(n) < 120)
temperature[heatwave_mask] += 9

humidity = 70 - (temperature-26)*1.5 + np.random.normal(0,3,n)
humidity = np.clip(humidity, 25, 95)

light = np.maximum(0, 800*np.sin(np.pi*(hours-6)/12)) + np.random.normal(0,20,n)
light = np.clip(light, 0, 1000)

water_level = np.zeros(n)
water_level[0] = 90
pump_status = np.zeros(n, dtype=int)

for i in range(1, n):
    decay = 0.6 + (temperature[i]-26)*0.05
    soil_moisture[i] = soil_moisture[i-1] - decay + np.random.normal(0,0.3)

    if soil_moisture[i] < THRESH['soil_moisture_low'] and water_level[i-1] > 10:
        pump_status[i] = 1
        soil_moisture[i] += 8
        water_level[i] = water_level[i-1] - 2
    else:
        water_level[i] = water_level[i-1] - 0.05

    soil_moisture[i] = np.clip(soil_moisture[i], 5, 70)
    water_level[i] = np.clip(water_level[i], 0, 100)

    # Refill on day 5 morning (index 192)
    if i == 192:
        water_level[i] = 90

# Force a low-water dip near end (days 6-7) to trigger alert
low_water_mask = (np.arange(n) >= 280) & (np.arange(n) < 300)
water_level[low_water_mask] = np.linspace(20, 8, low_water_mask.sum())

df = pd.DataFrame({
    'timestamp': timestamps,
    'soil_moisture_pct': soil_moisture.round(1),
    'temperature_c': temperature.round(1),
    'humidity_pct': humidity.round(1),
    'light_lux': light.round(0),
    'water_level_pct': water_level.round(1),
    'pump_status': pump_status,
})
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

df['alert_low_moisture'] = (df['soil_moisture_pct'] < THRESH['soil_moisture_low']).astype(int)
df['alert_high_temp']    = (df['temperature_c'] > THRESH['temp_high']).astype(int)
df['alert_low_humidity'] = (df['humidity_pct'] < THRESH['humidity_low']).astype(int)
df['alert_low_water']    = (df['water_level_pct'] < THRESH['water_level_low']).astype(int)
df['any_alert'] = df[['alert_low_moisture','alert_high_temp','alert_low_humidity','alert_low_water']].max(axis=1)

df.to_csv('sensor_data.csv', index=False)
print(f"✅ Generated {len(df)} sensor readings over 7 days (every 30 min)")
print(f"💧 Pump activated {df['pump_status'].sum()} times")
print(f"⚠️  Total alerts triggered: {df['any_alert'].sum()}")
print(f"   - Low moisture: {df['alert_low_moisture'].sum()}")
print(f"   - High temp: {df['alert_high_temp'].sum()}")
print(f"   - Low humidity: {df['alert_low_humidity'].sum()}")
print(f"   - Low water: {df['alert_low_water'].sum()}")

# CHART 1 — MAIN DASHBOARD
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('#0B1410')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0,0]); ax1.set_facecolor('#121F18')
ax1.plot(df['timestamp'], df['soil_moisture_pct'], color=GREEN, linewidth=1.5, label='Soil Moisture')
ax1.axhline(y=THRESH['soil_moisture_low'], color=RED, linestyle='--', linewidth=1.2, alpha=0.7, label='Low Threshold')
pump_on = df[df['pump_status']==1]
ax1.scatter(pump_on['timestamp'], pump_on['soil_moisture_pct'], color=BLUE, s=25, zorder=5, label='Pump ON', alpha=0.8)
ax1.set_title('🌱  Soil Moisture & Irrigation Events')
ax1.set_ylabel('Moisture (%)')
ax1.legend(framealpha=0.15, labelcolor='white', fontsize=8)
ax1.grid(True, alpha=0.25)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='x', rotation=25)

ax2 = fig.add_subplot(gs[0,1]); ax2.set_facecolor('#121F18')
ax2.plot(df['timestamp'], df['temperature_c'], color=ORANGE, linewidth=1.5, label='Temperature (°C)')
ax2.axhline(y=THRESH['temp_high'], color=RED, linestyle='--', linewidth=1, alpha=0.6, label='High Temp Threshold')
ax2b = ax2.twinx()
ax2b.plot(df['timestamp'], df['humidity_pct'], color=BLUE, linewidth=1.2, alpha=0.7, label='Humidity (%)')
ax2b.set_ylabel('Humidity (%)', color=BLUE)
ax2b.tick_params(axis='y', colors=BLUE)
ax2.set_title('🌡️  Temperature & Humidity (Heatwave Day 3)')
ax2.set_ylabel('Temperature (°C)', color=ORANGE)
ax2.tick_params(axis='y', colors=ORANGE)
ax2.legend(loc='upper left', framealpha=0.15, labelcolor='white', fontsize=8)
ax2.grid(True, alpha=0.25)
ax2.spines['top'].set_visible(False)
ax2.tick_params(axis='x', rotation=25)

ax3 = fig.add_subplot(gs[1,0]); ax3.set_facecolor('#121F18')
ax3.plot(df['timestamp'], df['light_lux'], color=YELLOW, linewidth=1.5)
ax3.fill_between(df['timestamp'], df['light_lux'], alpha=0.15, color=YELLOW)
ax3.set_title('☀️  Light Intensity (Lux)')
ax3.set_ylabel('Lux')
ax3.grid(True, alpha=0.25)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
ax3.tick_params(axis='x', rotation=25)

ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor('#121F18')
ax4.plot(df['timestamp'], df['water_level_pct'], color=BLUE, linewidth=1.5)
ax4.fill_between(df['timestamp'], df['water_level_pct'], alpha=0.15, color=BLUE)
ax4.axhline(y=THRESH['water_level_low'], color=RED, linestyle='--', linewidth=1.2, alpha=0.7, label='Low Water Threshold')
ax4.set_title('🚰  Water Tank Level (Refill Day 5)')
ax4.set_ylabel('Level (%)')
ax4.legend(framealpha=0.15, labelcolor='white', fontsize=8)
ax4.grid(True, alpha=0.25)
ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
ax4.tick_params(axis='x', rotation=25)

fig.suptitle('🌾  Smart Agriculture Monitoring — 7-Day Sensor Dashboard', fontsize=15, color='#E8F5ED', y=1.01)
plt.savefig('chart1_sensor_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0B1410')
plt.close(); print("Saved chart1")

# CHART 2 — DAILY SUMMARY
daily = df.groupby('date').agg(
    avg_moisture=('soil_moisture_pct','mean'),
    avg_temp=('temperature_c','mean'),
    avg_humidity=('humidity_pct','mean'),
    pump_runs=('pump_status','sum'),
    alerts=('any_alert','sum')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(daily)); w = 0.25
axes[0].bar(x-w, daily['avg_moisture'], w, label='Avg Moisture %', color=GREEN, alpha=0.85)
axes[0].bar(x,   daily['avg_temp'],     w, label='Avg Temp °C',    color=ORANGE, alpha=0.85)
axes[0].bar(x+w, daily['avg_humidity'], w, label='Avg Humidity %', color=BLUE, alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels([str(d)[5:] for d in daily['date']])
axes[0].set_title('📊  Daily Averages — Moisture, Temp, Humidity')
axes[0].legend(framealpha=0.15, labelcolor='white', fontsize=9)
axes[0].grid(True, axis='y', alpha=0.25)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

bars_p = axes[1].bar(x-0.15, daily['pump_runs'], 0.3, label='Pump Activations', color=BLUE, alpha=0.85)
bars_a = axes[1].bar(x+0.15, daily['alerts'],   0.3, label='Alerts Triggered', color=RED, alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels([str(d)[5:] for d in daily['date']])
axes[1].set_title('🚨  Daily Pump Activations & Alerts')
axes[1].legend(framealpha=0.15, labelcolor='white', fontsize=9)
axes[1].grid(True, axis='y', alpha=0.25)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
for b in bars_p:
    h=b.get_height()
    if h>0: axes[1].text(b.get_x()+b.get_width()/2, h+0.1, f'{int(h)}', ha='center', color='#D4F0DD', fontsize=9)
for b in bars_a:
    h=b.get_height()
    if h>0: axes[1].text(b.get_x()+b.get_width()/2, h+0.1, f'{int(h)}', ha='center', color='#D4F0DD', fontsize=9)

plt.tight_layout()
plt.savefig('chart2_daily_summary.png', dpi=150, bbox_inches='tight', facecolor='#0B1410')
plt.close(); print("Saved chart2")

# CHART 3 — HOURLY HEATMAP
hourly_pivot = df.pivot_table(values='temperature_c', index='date', columns='hour', aggfunc='mean')
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(hourly_pivot, annot=False, cmap='YlOrRd', linewidths=0.3, linecolor='#0B1410',
            cbar_kws={'label':'Temperature (°C)'}, ax=ax)
ax.set_title('🗓️  Hourly Temperature Pattern (Day × Hour) — Heatwave Visible on Day 3', fontsize=12, pad=12)
ax.set_xlabel('Hour of Day'); ax.set_ylabel('Date')
plt.tight_layout()
plt.savefig('chart3_hourly_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0B1410')
plt.close(); print("Saved chart3")

# CHART 4 — ALERT BREAKDOWN
alert_counts = {
    'Low Soil Moisture': df['alert_low_moisture'].sum(),
    'High Temperature': df['alert_high_temp'].sum(),
    'Low Humidity': df['alert_low_humidity'].sum(),
    'Low Water Level': df['alert_low_water'].sum(),
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
names = list(alert_counts.keys()); vals = list(alert_counts.values())
colors_alert = [RED, ORANGE, YELLOW, BLUE]
bars = ax1.barh(names, vals, color=colors_alert, alpha=0.85, edgecolor='none')
ax1.set_title('🚨  Total Alert Counts (7 Days)')
ax1.set_xlabel('Number of Alerts')
ax1.grid(True, axis='x', alpha=0.25)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
for bar, val in zip(bars, vals):
    ax1.text(val+0.3, bar.get_y()+bar.get_height()/2, str(val), va='center', color='#D4F0DD', fontsize=10)

status_counts = {'Normal': (df['any_alert']==0).sum(), 'Alert Active': (df['any_alert']==1).sum()}
wedges, texts, autotexts = ax2.pie(
    status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%',
    colors=[GREEN, RED], startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='#0B1410', linewidth=2),
    textprops={'color':'#D4F0DD','fontsize':10})
for at in autotexts: at.set_color('#0B1410'); at.set_fontweight('bold')
ax2.set_title('🟢  System Health Status')
plt.tight_layout()
plt.savefig('chart4_alerts.png', dpi=150, bbox_inches='tight', facecolor='#0B1410')
plt.close(); print("Saved chart4")

# LIVE MONITORING SIMULATION
print("\n" + "="*70)
print("  LIVE SENSOR MONITORING — LAST 10 READINGS")
print("="*70)
print(f"{'Time':<8} {'Moisture':>9} {'Temp':>7} {'Humidity':>9} {'Light':>7} {'Water':>7} {'Pump':>6} {'Alert'}")
print("-"*70)
for _, row in df.tail(10).iterrows():
    alert_str = "⚠️ YES" if row['any_alert'] else "✅ OK"
    pump_str  = "🟢 ON" if row['pump_status'] else "⚪ OFF"
    print(f"{row['timestamp'].strftime('%H:%M'):<8} {row['soil_moisture_pct']:>8.1f}% {row['temperature_c']:>6.1f}° "
          f"{row['humidity_pct']:>8.1f}% {row['light_lux']:>6.0f} {row['water_level_pct']:>6.1f}% {pump_str:>8} {alert_str}")
print("="*70)

# FINAL REPORT
uptime_pct = (df['any_alert']==0).mean()*100
print()
print("╔══════════════════════════════════════════════════════╗")
print("║   SMART AGRICULTURE MONITORING — FINAL REPORT       ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  📅 Monitoring Period : 7 Days (336 readings)       ║")
print(f"║  🌱 Avg Soil Moisture : {df['soil_moisture_pct'].mean():.1f}%{'':<24}║")
print(f"║  🌡️  Avg Temperature   : {df['temperature_c'].mean():.1f}°C{'':<23}║")
print(f"║  💧 Avg Humidity      : {df['humidity_pct'].mean():.1f}%{'':<24}║")
print(f"║  ☀️  Avg Light         : {df['light_lux'].mean():.0f} lux{'':<22}║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  💦 Pump Activations  : {int(df['pump_status'].sum()):<28}║")
print(f"║  🚨 Total Alerts      : {int(df['any_alert'].sum()):<28}║")
print(f"║  ✅ System Uptime     : {uptime_pct:.1f}%{'':<24}║")
print("╠══════════════════════════════════════════════════════╣")
print("║  📁 Files Saved:                                     ║")
print("║     sensor_data.csv                                 ║")
print("║     chart1_sensor_dashboard.png                     ║")
print("║     chart2_daily_summary.png                        ║")
print("║     chart3_hourly_heatmap.png                       ║")
print("║     chart4_alerts.png                               ║")
print("╚══════════════════════════════════════════════════════╝")
