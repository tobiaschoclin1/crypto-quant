import flet as ft
import requests
import time
import threading

# --- TU IP LOCAL ---
API_URL = "http://192.168.0.23:8000/analisis"

def main(page: ft.Page):
    page.title = "Crypto Quant Pro"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.bgcolor = "#0b1220" 

    running = True

    # --- UI ---
    lbl_precio = ft.Text("---", size=45, weight=ft.FontWeight.BOLD, color="white")
    lbl_decision = ft.Text("Sincronizando...", size=16, weight=ft.FontWeight.W_500, color="grey")
    
    header_container = ft.Container(
        content=ft.Column([
            ft.Text("BTC/USDT", size=14, color="grey"),
            lbl_precio,
            lbl_decision
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
        padding=ft.Padding(0, 50, 0, 20),
        alignment=ft.Alignment(0, 0)
    )

    # GRÁFICO (SOLUCIÓN DEFINITIVA)
    # 1. Usamos 'src' (universal).
    # 2. Usamos strings para 'fit' ("contain").
    # 3. Quitamos gapless_playback por si acaso.
    img_chart = ft.Image(
        src="", 
        width=400, 
        height=200, 
        fit="contain" 
    )

    chart_container = ft.Container(
        content=img_chart,
        height=200,
        alignment=ft.Alignment(0, 0),
        padding=ft.Padding(0, 10, 0, 20),
    )

    score_ring = ft.ProgressRing(width=80, height=80, stroke_width=8, value=0, color="grey")
    lbl_score = ft.Text("0", size=24, weight=ft.FontWeight.BOLD)
    
    score_stack = ft.Stack(
        [score_ring, lbl_score],
        alignment=ft.Alignment(0, 0),
        width=80, height=80
    )

    lv_detalles = ft.Column(spacing=10)

    def actualizar_ui():
        while running:
            try:
                r = requests.get(API_URL, timeout=5)
                data = r.json()
                
                precio = data['precio']
                score = data['score']
                decision = data['decision']
                img_b64 = data['grafico_img'] # Ahora viene con "data:image..."

                lbl_precio.value = f"${precio:,.2f}"
                lbl_decision.value = decision.upper()
                
                tema_color = "grey"
                if score >= 8: tema_color = "green"
                elif score <= 3: tema_color = "red"
                else: tema_color = "orange"
                
                lbl_decision.color = tema_color
                lbl_precio.color = "white"
                
                lbl_score.value = str(score)
                score_ring.value = score / 10
                score_ring.color = tema_color

                # ACTUALIZAR GRÁFICO
                if img_b64:
                    img_chart.src = img_b64
                    img_chart.update() # Forzamos repintado de imagen

                lv_detalles.controls.clear()
                for det in data['detalles']:
                    icono_nombre = ft.Icons.CHECK if "✅" in str(det) else ft.Icons.WARNING_AMBER
                    icono = ft.Icon(icono_nombre, color="grey", size=16)
                    txt = str(det).replace("✅", "").replace("❌", "").replace("⚠️", "").strip()
                    
                    lv_detalles.controls.append(
                        ft.Container(
                            content=ft.Row([icono, ft.Text(txt, size=12, color="#cbd5e1", expand=True)], alignment=ft.MainAxisAlignment.START),
                            padding=12,
                            bgcolor="#111827",
                            border_radius=8
                        )
                    )

                page.update()
                
            except Exception as e:
                print(f"Error: {e}")
                lbl_decision.value = "..."
                page.update()
            
            time.sleep(10)

    page.add(
        ft.Column([
            header_container,
            chart_container,
            ft.Container(
                content=ft.Row([
                    score_stack,
                    ft.Container(width=20),
                    ft.Column([
                        ft.Text("ANÁLISIS QUANT", size=12, weight=ft.FontWeight.BOLD, color="grey"),
                        lv_detalles
                    ], expand=True)
                ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START),
                padding=20,
                bgcolor="#0b1220",
                expand=True
            )
        ], spacing=0, expand=True)
    )

    t = threading.Thread(target=actualizar_ui, daemon=True)
    t.start()

if __name__ == "__main__":
    ft.app(main, view=ft.AppView.WEB_BROWSER, port=8550, host="0.0.0.0")