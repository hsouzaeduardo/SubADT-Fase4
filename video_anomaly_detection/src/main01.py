"""
Script Principal - Sistema de Análise de Vídeo
Integra detecção, rastreamento, classificação e geração de relatórios
"""

import cv2
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import timedelta

from video_anomaly_detection.src.detector01 import ObjectDetector
from video_anomaly_detection.src.activity_classifier01 import ActivityClassifier
from video_anomaly_detection.src.anomaly_detector01 import AnomalyDetector
from report_generator import ReportGenerator


class VideoAnalyzer:
    """Analisador completo de vídeo"""
    
    def __init__(self, confidence: float = 0.5, output_dir: str = 'output'):
        """
        Inicializa analisador
        
        Args:
            confidence: Confiança mínima para detecção
            output_dir: Diretório de saída
        """
        self.detector = ObjectDetector(confidence=confidence)
        self.classifier = None  # Será inicializado com FPS do vídeo
        self.anomaly_detector = None  # Será inicializado com FPS do vídeo
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dados de análise
        self.analysis_data = {
            'frames_data': [],
            'detection_stats': {'frames': [], 'object_counts': []},
            'tracks_info': {}
        }
    
    def analyze_video(self, video_path: str, show_video: bool = False, 
                     save_video: bool = True, generate_report: bool = True):
        """
        Analisa vídeo completo
        
        Args:
            video_path: Caminho do vídeo
            show_video: Exibir processamento em tempo real
            save_video: Salvar vídeo processado
            generate_report: Gerar relatório PDF
        """
        print(f"\n{'='*60}")
        print(f"ANÁLISE DE VÍDEO - Sistema de Detecção de Anomalias")
        print(f"{'='*60}\n")
        
        # Abrir vídeo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir o vídeo {video_path}")
            return
        
        # Propriedades do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Vídeo: {os.path.basename(video_path)}")
        print(f"Resolução: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Duração: {timedelta(seconds=int(duration))}")
        print(f"Total de frames: {total_frames}\n")
        
        # Inicializar classificadores
        self.classifier = ActivityClassifier(frame_rate=fps)
        self.anomaly_detector = AnomalyDetector(frame_rate=fps)
        
        # Configurar writer de vídeo
        video_writer = None
        if save_video:
            output_video_path = os.path.join(
                self.output_dir,
                f"processed_{os.path.basename(video_path)}"
            )
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (width, height)
            )
        
        # Processar vídeo
        frame_number = 0
        start_time = time.time()
        
        print("🔄 Processando vídeo...")
        progress_bar = tqdm(total=total_frames, desc="Frames", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_number / fps
            
            # 1. Detecção e rastreamento
            frame_annotated, tracks = self.detector.detect_and_track(frame)
            
            # 2. Classificação de atividades
            activities = self.classifier.classify(tracks, frame_number)
            
            # 3. Detecção de anomalias
            anomalies = self.anomaly_detector.detect(
                tracks, activities, frame_number, timestamp
            )
            
            # 4. Anotar frame com informações adicionais
            frame_annotated = self._annotate_frame(
                frame_annotated, tracks, activities, anomalies, frame_number, timestamp
            )
            
            # 5. Salvar dados
            self._save_frame_data(frame_number, tracks, activities, anomalies)
            
            # 6. Exibir/salvar
            if show_video:
                cv2.imshow('Video Analysis', frame_annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  Análise interrompida pelo usuário")
                    break
            
            if video_writer:
                video_writer.write(frame_annotated)
            
            frame_number += 1
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Liberar recursos
        cap.release()
        if video_writer:
            video_writer.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Tempo de processamento
        elapsed_time = time.time() - start_time
        processing_fps = frame_number / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✅ Processamento concluído!")
        print(f"⏱️  Tempo de processamento: {timedelta(seconds=int(elapsed_time))}")
        print(f"🚀 FPS de processamento: {processing_fps:.2f}")
        print(f"{'='*60}\n")
        
        # Compilar estatísticas
        self._compile_statistics(video_path, fps, duration, total_frames)
        
        # Gerar relatório
        if generate_report:
            print("📄 Gerando relatório...")
            report_gen = ReportGenerator(self.output_dir)
            
            # Salvar JSON
            json_path = report_gen.save_json_data(self.analysis_data)
            print(f"   💾 Dados JSON salvos em: {json_path}")
            
            # Gerar PDF
            pdf_path = report_gen.generate_report(video_path, self.analysis_data)
            print(f"   📋 Relatório PDF gerado: {pdf_path}")
        
        # Resumo
        self._print_summary()
        
        print(f"\n{'='*60}")
        print(f"📁 Arquivos de saída salvos em: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _annotate_frame(self, frame: np.ndarray, tracks: list, 
                       activities: dict, anomalies: list,
                       frame_number: int, timestamp: float) -> np.ndarray:
        """Adiciona anotações extras ao frame"""
        height, width = frame.shape[:2]
        
        # Overlay de informações
        overlay = frame.copy()
        
        # Painel de informações (topo)
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Informações do frame
        time_str = str(timedelta(seconds=int(timestamp)))
        cv2.putText(frame, f"Frame: {frame_number} | Tempo: {time_str}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Objetos Rastreados: {len(tracks)}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Contador de anomalias
        if anomalies:
            cv2.putText(frame, f"Anomalias: {len(anomalies)}", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Anotar atividades nos tracks
        for track in tracks:
            track_id = track['id']
            activity = activities.get(track_id, 'DESCONHECIDO')
            
            x1, y1, x2, y2 = map(int, track['bbox'])
            
            # Label da atividade
            activity_label = activity.replace('_', ' ')
            label_y = y2 + 20
            
            # Background do label
            label_size, _ = cv2.getTextSize(
                activity_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame, 
                         (x1, label_y - label_size[1] - 5),
                         (x1 + label_size[0], label_y),
                         (0, 255, 0), -1)
            
            cv2.putText(frame, activity_label, (x1, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Marcar anomalias
        for anomaly in anomalies:
            location = anomaly.get('location')
            severity = anomaly.get('severity', 'MEDIA')
            
            if location:
                x, y = map(int, location)
                
                # Cor baseada na severidade
                color = self.anomaly_detector.get_severity_color(severity)
                
                # Círculo de alerta
                cv2.circle(frame, (x, y), 30, color, 3)
                cv2.circle(frame, (x, y), 5, color, -1)
                
                # Label
                label = f"{anomaly['type']} ({severity})"
                cv2.putText(frame, label, (x + 35, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def _save_frame_data(self, frame_number: int, tracks: list, 
                        activities: dict, anomalies: list):
        """Salva dados do frame para análise posterior"""
        frame_data = {
            'frame': frame_number,
            'tracks_count': len(tracks),
            'tracks': [
                {
                    'id': t['id'],
                    'class': t.get('class_id'),
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
        """Compila estatísticas finais"""
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
        """Imprime resumo da análise"""
        summary = self.analysis_data['summary']
        activity_stats = self.analysis_data['activity_stats']
        anomaly_stats = self.analysis_data['anomaly_stats']
        
        print("\n" + "="*60)
        print("📊 RESUMO DA ANÁLISE")
        print("="*60)
        
        print(f"\n🎯 Detecção:")
        print(f"   • Total de objetos/pessoas rastreados: {summary['total_tracks']}")
        
        print(f"\n🏃 Atividades:")
        activity_counts = activity_stats.get('activity_counts', {})
        for activity, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {activity.replace('_', ' ')}: {count}")
        
        print(f"\n⚠️  Anomalias:")
        print(f"   • Total: {summary['total_anomalies']}")
        by_severity = anomaly_stats.get('by_severity', {})
        for severity in ['ALTA', 'MEDIA', 'BAIXA']:
            count = by_severity.get(severity, 0)
            if count > 0:
                emoji = {'ALTA': '🔴', 'MEDIA': '🟡', 'BAIXA': '🟢'}[severity]
                print(f"   {emoji} {severity}: {count}")
        
        print(f"\n🔍 Tipos de anomalias:")
        by_type = anomaly_stats.get('by_type', {})
        for anom_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {anom_type.replace('_', ' ')}: {count}")


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Sistema de Análise de Vídeo com Detecção de Anomalias',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Análise básica
  python main.py --input video.mp4

  # Análise com visualização em tempo real
  python main.py --input video.mp4 --show-video

  # Análise completa com todos os outputs
  python main.py --input video.mp4 --show-video --save-video --generate-report

  # Ajustar confiança de detecção
  python main.py --input video.mp4 --confidence 0.7
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Caminho do vídeo de entrada')
    parser.add_argument('--output', type=str, default='data/output',
                       help='Diretório de saída (padrão: data/output)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confiança mínima para detecção (0-1, padrão: 0.5)')
    parser.add_argument('--show-video', action='store_true',
                       help='Exibir processamento em tempo real')
    parser.add_argument('--save-video', action='store_true',
                       help='Salvar vídeo processado')
    parser.add_argument('--generate-report', action='store_true',
                       help='Gerar relatório PDF')
    
    args = parser.parse_args()
    
    # Verificar entrada
    if not os.path.exists(args.input):
        print(f"❌ Erro: Vídeo não encontrado: {args.input}")
        sys.exit(1)
    
    # Criar analisador
    analyzer = VideoAnalyzer(
        confidence=args.confidence,
        output_dir=args.output
    )
    
    # Analisar vídeo
    try:
        analyzer.analyze_video(
            video_path=args.input,
            show_video=args.show_video,
            save_video=args.save_video,
            generate_report=args.generate_report
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Análise interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro durante análise: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()