import sys
import glob
from analyzer.pipeline import Analyzer
from report import print_header, format_row, print_footer

def main():
    try:
        analyzer = Analyzer(
            vad_path="models/silero_vad.onnx",
            model_id="garystafford/wav2vec2-deepfake-voice-detector"
        )
        print(f"[*] Inference device: {str(analyzer.device).upper()}", file=sys.stderr)
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        return 2

    files = glob.glob("audio/*")
    if not files:
        print("Directory audio/ is empty.", file=sys.stderr)
        return 0

    print_header()
    
    for path in sorted(files):
        try:
            res = analyzer.analyze(path)
            print(format_row(res))
        except Exception as e:
            print(f"│ Error analyzing {path}: {e}")
            
    print_footer()

if __name__ == "__main__":
    sys.exit(main())