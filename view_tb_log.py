import argparse
from pathlib import Path
try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
except ImportError:
    try:
        # Альтернативный импорт для новых версий TensorFlow
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        HAS_EVENT_ACCUMULATOR = True
    except ImportError:
        HAS_EVENT_ACCUMULATOR = False
    HAS_TF = False
else:
    HAS_TF = True

def read_with_summary_iterator(log_file):
    """Чтение лога с использованием summary_iterator (низкоуровневый способ)."""
    print(f"Reading log file: {log_file}")
    try:
        for summary in summary_iterator(str(log_file)):
            if summary.step == 0 and len(summary.summary.value) == 0:
                continue  # Пропускаем метаданные
            for value in summary.summary.value:
                if value.HasField('simple_value'):
                    print(f"Step: {summary.step}, Tag: {value.tag}, Value: {value.simple_value}")
                elif value.HasField('tensor'):
                    # Для простоты не обрабатываем тензоры
                    print(f"Step: {summary.step}, Tag: {value.tag}, Value: <Tensor>")
    except Exception as e:
        print(f"Error reading log file: {e}")

def read_with_event_accumulator(log_file):
    """Чтение лога с использованием EventAccumulator (более высокоуровневый способ)."""
    print(f"Reading log file: {log_file}")
    try:
        event_acc = EventAccumulator(str(log_file))
        event_acc.Reload()
        # Получаем все скаляры
        tags = event_acc.Tags()['scalars']
        for tag in tags:
            scalar_events = event_acc.Scalars(tag)
            for event in scalar_events:
                print(f"Step: {event.step}, Tag: {tag}, Value: {event.value}")
    except Exception as e:
        print(f"Error reading log file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Read TensorBoard log file and print contents to console.")
    parser.add_argument("log_file", type=str, help="Path to the TensorBoard event file (e.g., events.out.tfevents.*)")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: File {log_path} does not exist.")
        return

    if HAS_TF:
        read_with_summary_iterator(log_path)
    elif HAS_EVENT_ACCUMULATOR:
        read_with_event_accumulator(log_path)
    else:
        print("Error: Neither TensorFlow nor TensorBoard is installed. Please install one of them.")
        print("Try: pip install tensorflow   or   pip install tensorboard")

if __name__ == "__main__":
    main()
