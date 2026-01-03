"""
Script Principal - Sistema de Análise de Vídeo (Revisado)
Integra detecção, rastreamento, classificação e geração de relatórios

Revisões aplicadas (foco em robustez e coerência temporal):
1) Remove dependência de `time.time()` para métricas de processamento (usa perf_counter).
2) Passa frame_number e timestamp para o detector/tracker (compatível com tracker revisitado).
3) Tratamento de FPS inválido (0/NaN) com fallback seguro.
4) Progress bar sem total quando CAP_PROP_FRAME_COUNT falha (ex.: streams).
5) Evita desenhar labels fora do frame (y negativo).
6) Armazena confidence no frame_data (útil para análises posteriores).
7) Corrige prints e remove emojis (ambiente corporativo/CLI).
"""

import cv2
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import timedelta  
from time import perf_counter

from detector import ObjectDetector
from activity_classifier import SoccerActivityClassifier
from anomaly_detector01 import AnomalyDetector
from report_generator import ReportGenerator


class VideoAnalyzer:
    """Analisador completo de vídeo"""

    def __init__(self, confidence: float = 0.5, output_dir: str = 'output'):
        self.detector = ObjectDetector(confidence=confidence)
        self.classifier = None
        self.anomaly_detector = None
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.analysis_data = {
            'frames_data': [],
            'detection_stats': {'frames': [], 'object_counts': []},
            'tracks_info': {}
        }

    def analyze_video(self, video_path: str, show_video: bool = False,
                      save_video: bool = True, generate_report: bool = True):

        print("\n" + "=" * 60)
        print("ANÁLISE DE VÍDEO - Sistema de Detecção e Classificação")
        print("=" * 60 + "\n")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir o vídeo: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or np.isnan(fps):
            # fallback seguro (comum em alguns codecs/streams)
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Duração estimada (pode ser 0 em streams)
        duration = (total_frames / fps) if total_frames > 0 else 0

        print(f"Vídeo: {os.path.basename(video_path)}")
        print(f"Resolução: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        if duration > 0:
            print(f"Duração: {timedelta(seconds=int(duration))}")
        if total_frames > 0:
            print(f"Total de frames: {total_frames}")
        print()

        # Inicializar módulos com FPS do vídeo
        self.classifier = SoccerActivityClassifier(frame_rate=fps)
        self.anomaly_detector = AnomalyDetector(frame_rate=fps)

        video_writer = None
        output_video_path = None
        if save_video:
            output_video_path = os.path.join(self.output_dir, f"processed_{os.path.basename(video_path)}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_number = 0
        start = perf_counter()

        # progress bar: se não tiver total_frames, tqdm trabalha sem total
        progress_bar = tqdm(total=total_frames if total_frames > 0 else None, desc="Frames", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Timestamp consistente (tempo do vídeo)
            timestamp = frame_number / fps

            # 1) Detecção e rastreamento
            # IMPORTANTE: seu detector revisitado deve aceitar (frame, frame_number, timestamp)
            try:
                frame_annotated, tracks = self.detector.detect_and_track(frame, frame_number, timestamp)
            except TypeError:
                # Compatibilidade com versão antiga do detector (sem frame_number/timestamp)
                frame_annotated, tracks = self.detector.detect_and_track(frame)

            # 2) Classificação de atividades
            activities = self.classifier.classify(tracks, frame_number)

            # 3) Detecção de anomalias (já corrigida para usar timestamp)
            anomalies = self.anomaly_detector.detect(tracks, activities, frame_number, timestamp)

            # 4) Anotar frame com painel e labels
            frame_annotated = self._annotate_frame(
                frame_annotated, tracks, activities, anomalies, frame_number, timestamp
            )

            # 5) Salvar dados do frame
            self._save_frame_data(frame_number, tracks, activities, anomalies)

            # 6) Exibir/salvar
            if show_video:
                cv2.imshow('Video Analysis', frame_annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nAnálise interrompida pelo usuário.")
                    break

            if video_writer:
                video_writer.write(frame_annotated)

            frame_number += 1
            progress_bar.update(1)

        progress_bar.close()

        cap.release()
        if video_writer:
            video_writer.release()
        if show_video:
            cv2.destroyAllWindows()

        elapsed = perf_counter() - start
        processing_fps = frame_number / elapsed if elapsed > 0 else 0.0

        print("\n" + "=" * 60)
        print("Processamento concluído.")
        print(f"Tempo de processamento: {timedelta(seconds=int(elapsed))}")
        print(f"FPS de processamento: {processing_fps:.2f}")
        if output_video_path:
            print(f"Vídeo processado salvo em: {output_video_path}")
        print("=" * 60 + "\n")

        self._compile_statistics(video_path, fps, duration, total_frames)

        if generate_report:
            print("Gerando relatório...")
            report_gen = ReportGenerator(self.output_dir)

            json_path = report_gen.save_json_data(self.analysis_data)
            print(f"Dados JSON salvos em: {json_path}")

            pdf_path = report_gen.generate_report(video_path, self.analysis_data)
            print(f"Relatório PDF gerado em: {pdf_path}")

        self._print_summary()

        print("\n" + "=" * 60)
        print(f"Arquivos de saída em: {self.output_dir}")
        print("=" * 60 + "\n")

    def _annotate_frame(self, frame: np.ndarray, tracks: list,
                        activities: dict, anomalies: list,
                        frame_number: int, timestamp: float) -> np.ndarray:

        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        time_str = str(timedelta(seconds=int(timestamp)))
        cv2.putText(frame, f"Frame: {frame_number} | Tempo: {time_str}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracks ativos: {len(tracks)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if anomalies:
            cv2.putText(frame, f"Alertas: {len(anomalies)}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Atividades por track
        for track in tracks:
            tid = track['id']
            activity = activities.get(tid, 'DESCONHECIDO')

            x1, y1, x2, y2 = map(int, track['bbox'])

            label = activity.replace('_', ' ')
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # tentar desenhar abaixo; se estourar, desenha acima
            y_label_bottom = y2 + 20
            y_label_top = y_label_bottom - th - 5

            if y_label_bottom >= h:
                # desenha acima do bbox
                y_label_bottom = max(0, y1 - 5)
                y_label_top = max(0, y_label_bottom - th - 5)

            x_label_right = min(w - 1, x1 + tw)
            x_label_left = max(0, x1)

            cv2.rectangle(frame,
                          (x_label_left, y_label_top),
                          (x_label_right, y_label_bottom),
                          (0, 255, 0), -1)

            cv2.putText(frame, label, (x_label_left, max(0, y_label_bottom - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Anomalias
        for anomaly in anomalies:
            loc = anomaly.get('location')
            if not loc:
                continue

            severity = anomaly.get('severity', 'MEDIA')
            x, y = map(int, loc)

            color = self.anomaly_detector.get_severity_color(severity)
            cv2.circle(frame, (x, y), 30, color, 3)
            cv2.circle(frame, (x, y), 5, color, -1)

            label = f"{anomaly['type']} ({severity})"
            cv2.putText(frame, label, (x + 35, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def _save_frame_data(self, frame_number: int, tracks: list,
                         activities: dict, anomalies: list):

        frame_data = {
            'frame': frame_number,
            'tracks_count': len(tracks),
            'tracks': [
                {
                    'id': t['id'],
                    'class_id': t.get('class_id'),
                    'confidence': t.get('confidence'),
                    'bbox': t['bbox'],
                    'activity': activities.get(t['id'])
                }
                for t in tracks
            ],
            'anomalies': [
                {
                    'type': a['type'],
                    'severity': a['severity'],
                    'location': a.get('location')
                }
                for a in anomalies
            ]
        }

        self.analysis_data['frames_data'].append(frame_data)
        self.analysis_data['detection_stats']['frames'].append(frame_number)
        self.analysis_data['detection_stats']['object_counts'].append(len(tracks))

    def _compile_statistics(self, video_path: str, fps: float,
                            duration: float, total_frames: int):

        activity_stats = self.classifier.get_activity_statistics()
        anomaly_stats = self.anomaly_detector.get_statistics()

        self.analysis_data.update({
            'video_path': video_path,
            'fps': fps,
            'duration': duration,
            'total_frames': total_frames,
            'activity_stats': activity_stats,
            'anomaly_stats': anomaly_stats,
            'summary': {
                'total_tracks': activity_stats.get('total_tracks', 0),
                'total_activities': len(activity_stats.get('activity_counts', {})),
                'total_anomalies': anomaly_stats.get('total_anomalies', 0),
                'high_severity_anomalies': anomaly_stats.get('by_severity', {}).get('ALTA', 0)
            }
        })

    def _print_summary(self):
        summary = self.analysis_data.get('summary', {})
        activity_stats = self.analysis_data.get('activity_stats', {})
        anomaly_stats = self.analysis_data.get('anomaly_stats', {})

        print("\n" + "=" * 60)
        print("RESUMO DA ANÁLISE")
        print("=" * 60)

        print("\nDetecção:")
        print(f"  - Total de tracks: {summary.get('total_tracks', 0)}")

        print("\nAtividades:")
        activity_counts = activity_stats.get('activity_counts', {})
        for activity, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {activity.replace('_', ' ')}: {count}")

        print("\nAlertas/Anomalias:")
        print(f"  - Total: {summary.get('total_anomalies', 0)}")
        by_sev = anomaly_stats.get('by_severity', {})
        for sev in ['ALTA', 'MEDIA', 'BAIXA']:
            c = by_sev.get(sev, 0)
            if c > 0:
                print(f"  - {sev}: {c}")

        print("\nTipos:")
        by_type = anomaly_stats.get('by_type', {})
        for t, c in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {t.replace('_', ' ')}: {c}")


def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Análise de Vídeo com Detecção de Anomalias',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input', type=str, required=True, help='Caminho do vídeo de entrada')
    parser.add_argument('--output', type=str, default='data/output', help='Diretório de saída')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confiança mínima (0-1)')
    parser.add_argument('--show-video', action='store_true', help='Exibir em tempo real')
    parser.add_argument('--save-video', action='store_true', help='Salvar vídeo processado')
    parser.add_argument('--generate-report', action='store_true', help='Gerar relatório PDF')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Erro: Vídeo não encontrado: {args.input}")
        sys.exit(1)

    analyzer = VideoAnalyzer(confidence=args.confidence, output_dir=args.output)

    try:
        analyzer.analyze_video(
            video_path=args.input,
            show_video=args.show_video,
            save_video=args.save_video,
            generate_report=args.generate_report
        )
    except KeyboardInterrupt:
        print("\nAnálise interrompida pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\nErro durante análise: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
