import logging
import queue
import threading
import time
import gradio as gr
from src.agents.deal_agent_framework import DealAgentFramework
from src.log_utils import reformat
import plotly.graph_objects as go


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def html_for(log_data):
    output = "<br>".join(log_data[-18:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #1e1e1e; padding: 10px; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; line-height: 1.6;">
    <style>
        #scrollContent {{
            color: #e0e0e0;
        }}
    </style>
    {output}
    </div>
    """


def setup_logging(log_queue):
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class App:
    def __init__(self):
        self.agent_framework = None
        self.last_run_time = None
        self.is_scanning = False

    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
            self.agent_framework.init_agents_as_needed()
        return self.agent_framework

    def run(self):
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
            log_data = gr.State([])

            def table_for(opps):
                return [
                    [
                        opp.deal.product_description,
                        f"${opp.deal.price:.2f}",
                        f"${opp.estimate:.2f}",
                        f"${opp.discount:.2f}",
                        opp.deal.url,
                    ]
                    for opp in opps
                ]

            def update_output(log_data, log_queue, result_queue):
                initial_result = table_for(self.get_agent_framework().memory)
                final_result = None
                while True:
                    try:
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message))
                        yield (
                            log_data,
                            html_for(log_data),
                            final_result or initial_result,
                        )
                    except queue.Empty:
                        try:
                            final_result = result_queue.get_nowait()
                            yield (
                                log_data,
                                html_for(log_data),
                                final_result or initial_result,
                            )
                        except queue.Empty:
                            if final_result is not None:
                                break
                            time.sleep(0.1)

            def do_run():
                if self.is_scanning:
                    logging.info("⚠️ Scan already in progress, skipping...")
                    return table_for(self.get_agent_framework().memory)

                self.is_scanning = True
                self.last_run_time = time.time()
                logging.info("🚀 Starting new scan cycle...")
                try:
                    new_opportunities = self.get_agent_framework().run()
                    table = table_for(new_opportunities)
                    logging.info(
                        f"✅ Scan cycle complete. Total opportunities in memory: {len(new_opportunities)}"
                    )
                    return table
                finally:
                    self.is_scanning = False

            def run_with_logging(initial_log_data):
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                setup_logging(log_queue)

                def worker():
                    try:
                        result = do_run()
                        result_queue.put(result)
                    except Exception as e:
                        logging.error(f"❌ Error during scan: {str(e)}")
                        import traceback

                        logging.error(traceback.format_exc())
                        result_queue.put(None)

                thread = threading.Thread(target=worker)
                thread.start()

                for log_data, output, final_result in update_output(
                    initial_log_data, log_queue, result_queue
                ):
                    yield log_data, output, final_result

            def update_countdown():
                """Update countdown timer every second"""
                if self.last_run_time is None:
                    return "⏳ First scan will start immediately after initialization"

                elapsed = time.time() - self.last_run_time
                remaining = max(0, 300 - int(elapsed))

                if remaining == 0:
                    return "🔄 Scanning for deals now..."

                minutes = remaining // 60
                seconds = remaining % 60
                return f"⏰ Next scan in: {minutes:02d}:{seconds:02d}"

            def manual_scan_trigger(initial_log_data):
                """Manual button to trigger a scan immediately"""
                logging.info("🔘 Manual scan triggered by user")
                for result in run_with_logging(initial_log_data):
                    yield result

            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:24px"><strong>The Price is Right</strong> - Autonomous Agent Framework that hunts for deals</div>'
                )
            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:14px">A proprietary fine-tuned LLM deployed on Modal and a RAG pipeline with a frontier model collaborate to send push notifications with great online deals.</div>'
                )

            # Countdown Timer and Manual Scan Button Row
            with gr.Row():
                with gr.Column(scale=3):
                    countdown_display = gr.Markdown(
                        value="⏳ Initializing...",
                        elem_classes="countdown-timer",
                    )
                with gr.Column(scale=1):
                    manual_scan_btn = gr.Button(
                        "🔍 Scan Now", variant="primary"
                    )

            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=[
                        "Deals found so far",
                        "Price",
                        "Estimate",
                        "Discount",
                        "URL",
                    ],
                    wrap=True,
                    column_widths=[6, 1, 1, 1, 3],
                    row_count=10,
                    col_count=5,
                    max_height=400,
                )
            with gr.Row():
                logs = gr.HTML()

            # Initial load - trigger first scan
            ui.load(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

            # Main timer for scanning (every 300 seconds = 5 minutes)
            scan_timer = gr.Timer(value=300, active=True)
            scan_timer.tick(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

            # Countdown display timer (updates every 1 second)
            countdown_timer = gr.Timer(value=1, active=True)
            countdown_timer.tick(update_countdown, outputs=[countdown_display])

            # Manual scan button
            manual_scan_btn.click(
                manual_scan_trigger,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

        ui.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    App().run()
