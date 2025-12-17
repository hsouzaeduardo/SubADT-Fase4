"""
Módulo de Geração de Relatórios
Cria relatórios em PDF com estatísticas e visualizações
"""

import os
import json
from datetime import datetime
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT


class ReportGenerator:
    """Gerador de relatórios de análise de vídeo"""
    
    def __init__(self, output_dir: str):
        """
        Inicializa gerador de relatórios
        
        Args:
            output_dir: Diretório para salvar relatórios
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Estilo
        sns.set_style("whitegrid")
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#06D6A0',
            'warning': '#F18F01',
            'danger': '#C73E1D'
        }
    
    def generate_report(self, video_path: str, analysis_data: Dict) -> str:
        """
        Gera relatório completo em PDF
        
        Args:
            video_path: Caminho do vídeo analisado
            analysis_data: Dados da análise
            
        Returns:
            Caminho do relatório gerado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(self.output_dir, f"relatorio_{timestamp}.pdf")
        
        # Criar documento
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Estilos customizados
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor(self.colors['primary']),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor(self.colors['primary']),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Título
        story.append(Paragraph("Relatório de Análise de Vídeo", title_style))
        story.append(Paragraph(f"Inteligência Artificial e Visão Computacional", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Informações do vídeo
        story.append(Paragraph("Informações do Vídeo", heading_style))
        video_info = [
            ["Vídeo:", os.path.basename(video_path)],
            ["Data de Análise:", datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
            ["Duração:", f"{analysis_data.get('duration', 0):.2f} segundos"],
            ["Total de Frames:", str(analysis_data.get('total_frames', 0))],
            ["FPS:", str(analysis_data.get('fps', 0))]
        ]
        
        table = Table(video_info, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor(self.colors['primary'])),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Resumo Executivo
        story.append(Paragraph("Resumo Executivo", heading_style))
        summary = analysis_data.get('summary', {})
        summary_text = f"""
        Durante a análise do vídeo, foram detectados <b>{summary.get('total_tracks', 0)}</b> objetos/pessoas únicos.
        O sistema identificou <b>{summary.get('total_activities', 0)}</b> atividades diferentes e
        detectou <b>{summary.get('total_anomalies', 0)}</b> eventos anômalos, dos quais
        <b>{summary.get('high_severity_anomalies', 0)}</b> foram classificados como alta severidade.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Estatísticas de Detecção
        story.append(Paragraph("Estatísticas de Detecção e Rastreamento", heading_style))
        detection_stats = analysis_data.get('detection_stats', {})
        
        if detection_stats:
            # Gráfico de objetos detectados ao longo do tempo
            fig_path = self._plot_detection_timeline(detection_stats)
            if fig_path and os.path.exists(fig_path):
                story.append(Image(fig_path, width=5*inch, height=3*inch))
                story.append(Spacer(1, 10))
        
        # Classificação de Atividades
        story.append(PageBreak())
        story.append(Paragraph("Classificação de Atividades", heading_style))
        
        activity_stats = analysis_data.get('activity_stats', {})
        if activity_stats:
            story.append(Paragraph("Atividades Detectadas:", styles['Heading3']))
            story.append(Spacer(1, 10))
            
            # Tabela de atividades
            activity_counts = activity_stats.get('activity_counts', {})
            activity_data = [["Atividade", "Ocorrências", "Percentual"]]
            total = sum(activity_counts.values())
            
            for activity, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                activity_data.append([
                    activity.replace('_', ' ').title(),
                    str(count),
                    f"{percentage:.1f}%"
                ])
            
            table = Table(activity_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['primary'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Gráfico de pizza
            fig_path = self._plot_activity_distribution(activity_counts)
            if fig_path and os.path.exists(fig_path):
                story.append(Image(fig_path, width=4*inch, height=4*inch))
        
        # Detecção de Anomalias
        story.append(PageBreak())
        story.append(Paragraph("Detecção de Anomalias", heading_style))
        
        anomaly_stats = analysis_data.get('anomaly_stats', {})
        if anomaly_stats:
            # Estatísticas gerais
            by_severity = anomaly_stats.get('by_severity', {})
            severity_data = [["Severidade", "Quantidade", "Cor"]]
            
            for severity in ['ALTA', 'MEDIA', 'BAIXA']:
                count = by_severity.get(severity, 0)
                color_map = {'ALTA': 'Vermelho', 'MEDIA': 'Amarelo', 'BAIXA': 'Verde'}
                severity_data.append([severity, str(count), color_map[severity]])
            
            table = Table(severity_data, colWidths=[2*inch, 2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['danger'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
            
            # Tipos de anomalias
            by_type = anomaly_stats.get('by_type', {})
            if by_type:
                story.append(Paragraph("Tipos de Anomalias Detectadas:", styles['Heading3']))
                story.append(Spacer(1, 10))
                
                type_data = [["Tipo", "Quantidade", "Descrição"]]
                for anom_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                    desc = self._get_anomaly_description(anom_type)
                    type_data.append([
                        anom_type.replace('_', ' ').title(),
                        str(count),
                        desc
                    ])
                
                table = Table(type_data, colWidths=[2*inch, 1*inch, 3*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['warning'])),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (1, -1), 'CENTER'),
                    ('ALIGN', (2, 0), (2, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP')
                ]))
                story.append(table)
                story.append(Spacer(1, 20))
                
                # Timeline de anomalias
                fig_path = self._plot_anomaly_timeline(anomaly_stats)
                if fig_path and os.path.exists(fig_path):
                    story.append(Image(fig_path, width=5*inch, height=3*inch))
        
        # Conclusões
        story.append(PageBreak())
        story.append(Paragraph("Conclusões e Insights", heading_style))
        
        conclusions = self._generate_conclusions(analysis_data)
        for conclusion in conclusions:
            story.append(Paragraph(f"• {conclusion}", styles['Normal']))
            story.append(Spacer(1, 8))
        
        # Gerar PDF
        doc.build(story)
        return pdf_path
    
    def _plot_detection_timeline(self, detection_stats: Dict) -> str:
        """Gera gráfico de linha da detecção ao longo do tempo"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Dados de exemplo (você deve adaptar conforme seus dados reais)
            frames = detection_stats.get('frames', [])
            counts = detection_stats.get('object_counts', [])
            
            if frames and counts:
                ax.plot(frames, counts, color=self.colors['primary'], linewidth=2)
                ax.fill_between(frames, counts, alpha=0.3, color=self.colors['primary'])
                
                ax.set_xlabel('Frame', fontsize=12)
                ax.set_ylabel('Número de Objetos Detectados', fontsize=12)
                ax.set_title('Objetos Detectados ao Longo do Tempo', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                output_path = os.path.join(self.output_dir, 'detection_timeline.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                return output_path
        except Exception as e:
            print(f"Erro ao gerar gráfico de timeline: {e}")
        
        return None
    
    def _plot_activity_distribution(self, activity_counts: Dict) -> str:
        """Gera gráfico de pizza da distribuição de atividades"""
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            labels = [k.replace('_', ' ').title() for k in activity_counts.keys()]
            sizes = list(activity_counts.values())
            colors_list = plt.cm.Set3(range(len(labels)))
            
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                colors=colors_list, startangle=90
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Distribuição de Atividades', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, 'activity_distribution.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Erro ao gerar gráfico de distribuição: {e}")
        
        return None
    
    def _plot_anomaly_timeline(self, anomaly_stats: Dict) -> str:
        """Gera gráfico de timeline das anomalias"""
        try:
            timeline = anomaly_stats.get('timeline', [])
            if not timeline:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Agrupar por tipo
            by_type = {}
            for item in timeline:
                anom_type = item['type']
                timestamp = item['timestamp']
                
                if anom_type not in by_type:
                    by_type[anom_type] = []
                by_type[anom_type].append(timestamp)
            
            # Plotar
            for i, (anom_type, timestamps) in enumerate(by_type.items()):
                ax.scatter(timestamps, [i] * len(timestamps), 
                          label=anom_type.replace('_', ' ').title(),
                          s=100, alpha=0.6)
            
            ax.set_xlabel('Tempo (segundos)', fontsize=12)
            ax.set_ylabel('Tipo de Anomalia', fontsize=12)
            ax.set_title('Timeline de Anomalias Detectadas', fontsize=14, fontweight='bold')
            ax.set_yticks(range(len(by_type)))
            ax.set_yticklabels([k.replace('_', ' ').title() for k in by_type.keys()])
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, 'anomaly_timeline.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Erro ao gerar timeline de anomalias: {e}")
        
        return None
    
    @staticmethod
    def _get_anomaly_description(anomaly_type: str) -> str:
        """Retorna descrição de um tipo de anomalia"""
        descriptions = {
            'MOVIMENTO_SUBITO': 'Aceleração repentina',
            'VELOCIDADE_ANORMAL': 'Velocidade excessiva',
            'PARADA_PROLONGADA': 'Imobilidade prolongada',
            'AGLOMERACAO': 'Múltiplas pessoas próximas',
            'MOVIMENTO_REVERSO': 'Retorno ao ponto inicial',
            'OBJETO_ABANDONADO': 'Objeto deixado no local',
            'DIRECAO_PROIBIDA': 'Movimento não esperado'
        }
        return descriptions.get(anomaly_type, 'Evento atípico')
    
    def _generate_conclusions(self, analysis_data: Dict) -> List[str]:
        """Gera conclusões automáticas baseadas nos dados"""
        conclusions = []
        
        summary = analysis_data.get('summary', {})
        activity_stats = analysis_data.get('activity_stats', {})
        anomaly_stats = analysis_data.get('anomaly_stats', {})
        
        # Análise de atividades
        activity_counts = activity_stats.get('activity_counts', {})
        if activity_counts:
            most_common = max(activity_counts, key=activity_counts.get)
            conclusions.append(
                f"A atividade mais frequente detectada foi '{most_common.replace('_', ' ')}', "
                f"representando {activity_counts[most_common]} ocorrências."
            )
        
        # Análise de anomalias
        total_anomalies = summary.get('total_anomalies', 0)
        high_severity = summary.get('high_severity_anomalies', 0)
        
        if total_anomalies > 0:
            if high_severity > 0:
                conclusions.append(
                    f"Foram detectadas {high_severity} anomalias de alta severidade que "
                    f"requerem atenção imediata."
                )
            else:
                conclusions.append(
                    "Nenhuma anomalia de alta severidade foi detectada. "
                    "O ambiente apresenta comportamento dentro do esperado."
                )
        
        # Padrões de movimentação
        if activity_counts.get('PARADO', 0) > activity_counts.get('CAMINHANDO', 0) * 2:
            conclusions.append(
                "Observou-se um padrão de baixa movimentação no ambiente, "
                "com muitas pessoas ou objetos permanecendo estáticos."
            )
        
        if activity_counts.get('CORRENDO', 0) > 0:
            conclusions.append(
                f"Foram detectados {activity_counts['CORRENDO']} eventos de corrida, "
                "o que pode indicar urgência ou situações incomuns."
            )
        
        # Interações
        if activity_counts.get('INTERAGINDO', 0) > 0:
            conclusions.append(
                f"Identificadas {activity_counts['INTERAGINDO']} interações entre pessoas, "
                "indicando um ambiente socialmente ativo."
            )
        
        return conclusions
    
    def save_json_data(self, data: Dict, filename: str = 'analysis_data.json'):
        """Salva dados da análise em JSON"""
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path


if __name__ == "__main__":
    # Teste do gerador
    generator = ReportGenerator('output/')
    
    # Dados de exemplo
    test_data = {
        'duration': 120.0,
        'total_frames': 3600,
        'fps': 30,
        'summary': {
            'total_tracks': 15,
            'total_activities': 5,
            'total_anomalies': 8,
            'high_severity_anomalies': 2
        },
        'activity_stats': {
            'activity_counts': {
                'CAMINHANDO': 450,
                'PARADO': 320,
                'CORRENDO': 45,
                'INTERAGINDO': 89
            }
        },
        'anomaly_stats': {
            'by_severity': {
                'ALTA': 2,
                'MEDIA': 3,
                'BAIXA': 3
            },
            'by_type': {
                'MOVIMENTO_SUBITO': 2,
                'VELOCIDADE_ANORMAL': 1,
                'PARADA_PROLONGADA': 3,
                'AGLOMERACAO': 2
            }
        }
    }
    
    pdf_path = generator.generate_report('test_video.mp4', test_data)
    print(f"Relatório gerado: {pdf_path}")