"""
Script de Verificação e Testes
Valida instalação e funcionalidade básica do sistema
"""

import sys
import subprocess
import importlib


def verificar_python():
    """Verifica versão do Python"""
    print("🔍 Verificando Python...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} é muito antigo")
        print(f"   📋 Requer Python 3.8 ou superior")
        return False


def verificar_dependencias():
    """Verifica se todas as dependências estão instaladas"""
    print("\n🔍 Verificando dependências...")
    
    dependencias_criticas = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('ultralytics', 'ultralytics'),
        ('torch', 'torch'),
        ('reportlab', 'reportlab'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy')
    ]
    
    todas_ok = True
    
    for modulo, pacote in dependencias_criticas:
        try:
            importlib.import_module(modulo)
            print(f"   ✅ {pacote} instalado")
        except ImportError:
            print(f"   ❌ {pacote} NÃO instalado")
            print(f"      Execute: pip install {pacote}")
            todas_ok = False
    
    return todas_ok


def verificar_gpu():
    """Verifica disponibilidade de GPU"""
    print("\n🔍 Verificando GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(1)
            gpu_count = torch.cuda.device_count()
            print(f"   ✅ GPU disponível: {gpu_name}")
            print(f"   📊 {gpu_count} GPU(s) detectada(s)")
            return True
        else:
            print(f"   ℹ️  GPU CUDA não disponível")
            print(f"   📋 Sistema funcionará em CPU (mais lento)")
            return False
    except Exception as e:
        print(f"   ⚠️  Erro ao verificar GPU: {e}")
        return False


def verificar_estrutura():
    """Verifica estrutura de diretórios"""
    print("\n🔍 Verificando estrutura de diretórios...")
    
    import os
    
    diretorios = [
        'src',
        'data/input',
        'data/output',
        'models'
    ]
    
    todos_ok = True
    
    for dir_path in diretorios:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}/ OK")
        else:
            print(f"   ❌ {dir_path}/ NÃO EXISTE")
            print(f"      Execute: mkdir -p {dir_path}")
            todos_ok = False
    
    return todos_ok


def verificar_modulos():
    """Verifica se os módulos do projeto podem ser importados"""
    print("\n🔍 Verificando módulos do projeto...")
    
    sys.path.insert(0, 'src')
    
    modulos = [
        'detector',
        'activity_classifier',
        'anomaly_detector',
        'report_generator'
    ]
    
    todos_ok = True
    
    for modulo in modulos:
        try:
            importlib.import_module(modulo)
            print(f"   ✅ {modulo}.py OK")
        except Exception as e:
            print(f"   ❌ {modulo}.py ERRO: {str(e)[:50]}")
            todos_ok = False
    
    return todos_ok


def teste_deteccao():
    """Teste básico de detecção"""
    print("\n🔍 Testando detector...")
    
    try:
        sys.path.insert(0, 'src')
        from detector import ObjectDetector
        import numpy as np
        
        # Criar detector
        detector = ObjectDetector()
        print("   ✅ Detector inicializado")
        
        # Criar frame de teste
        frame_test = np.zeros((640, 480, 3), dtype=np.uint8)
        
        # Tentar detectar
        frame_annotated, tracks = detector.detect_and_track(frame_test)
        print(f"   ✅ Detecção funcionando (encontrados {len(tracks)} objetos)")
        
        return True
    except Exception as e:
        print(f"   ❌ Erro no detector: {e}")
        return False


def teste_classificador():
    """Teste básico de classificador"""
    print("\n🔍 Testando classificador de atividades...")
    
    try:
        sys.path.insert(0, 'src')
        from activity_classifier import ActivityClassifier
        import numpy as np
        
        classifier = ActivityClassifier(frame_rate=30.0)
        print("   ✅ Classificador inicializado")
        
        # Criar track de teste
        test_track = {
            'id': 1,
            'bbox': [100, 100, 150, 200],
            'class_id': 0,
            'velocity': np.array([1.0, 0.5])
        }
        
        activities = classifier.classify([test_track], 0)
        print(f"   ✅ Classificação funcionando: {activities.get(1)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Erro no classificador: {e}")
        return False


def teste_detector_anomalias():
    """Teste básico de detector de anomalias"""
    print("\n🔍 Testando detector de anomalias...")
    
    try:
        sys.path.insert(0, 'src')
        from anomaly_detector import AnomalyDetector
        import numpy as np
        
        detector = AnomalyDetector(frame_rate=30.0)
        print("   ✅ Detector de anomalias inicializado")
        
        # Criar track e atividades de teste
        test_track = {
            'id': 1,
            'bbox': [100, 100, 150, 200],
            'class_id': 0,
            'velocity': np.array([15.0, 10.0])  # Velocidade alta
        }
        
        activities = {1: 'CORRENDO'}
        
        anomalies = detector.detect([test_track], activities, 0, 0.0)
        print(f"   ✅ Detecção de anomalias funcionando ({len(anomalies)} detectadas)")
        
        return True
    except Exception as e:
        print(f"   ❌ Erro no detector de anomalias: {e}")
        return False


def executar_verificacao_completa():
    """Executa verificação completa do sistema"""
    print("="*60)
    print("VERIFICAÇÃO DO SISTEMA DE ANÁLISE DE VÍDEO")
    print("="*60)
    
    resultados = {
        'Python': verificar_python(),
        'Dependências': verificar_dependencias(),
        'GPU': verificar_gpu(),
        'Estrutura': verificar_estrutura(),
        'Módulos': verificar_modulos(),
        'Detector': teste_deteccao(),
        'Classificador': teste_classificador(),
        'Anomalias': teste_detector_anomalias()
    }
    
    print("\n" + "="*60)
    print("RESUMO DA VERIFICAÇÃO")
    print("="*60)
    
    total = len(resultados)
    ok = sum(1 for v in resultados.values() if v)
    
    for componente, status in resultados.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {componente}")
    
    print(f"\n📊 Resultado: {ok}/{total} verificações passaram")
    
    if ok == total:
        print("\n🎉 Sistema pronto para uso!")
        print("\n📝 Próximos passos:")
        print("   1. Baixe o vídeo de teste")
        print("   2. Coloque em data/input/")
        print("   3. Execute: python src/main.py --input data/input/seu_video.mp4")
    else:
        print("\n⚠️  Sistema com problemas. Corrija os erros acima.")
        print("\n💡 Dica: Execute 'pip install -r requirements.txt' novamente")
    
    print("\n" + "="*60)
    
    return ok == total


def menu_testes():
    """Menu interativo de testes"""
    while True:
        print("\n" + "="*60)
        print("MENU DE TESTES")
        print("="*60)
        print("\n1. Verificação completa do sistema")
        print("2. Verificar apenas dependências")
        print("3. Teste de detecção")
        print("4. Teste de classificação")
        print("5. Teste de anomalias")
        print("0. Sair")
        print("\n" + "="*60)
        
        escolha = input("\nEscolha uma opção: ")
        
        if escolha == '1':
            executar_verificacao_completa()
        elif escolha == '2':
            verificar_dependencias()
        elif escolha == '3':
            teste_deteccao()
        elif escolha == '4':
            teste_classificador()
        elif escolha == '5':
            teste_detector_anomalias()
        elif escolha == '0':
            print("\nSaindo...")
            break
        else:
            print("\n❌ Opção inválida!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Modo automático
        sucesso = executar_verificacao_completa()
        sys.exit(0 if sucesso else 1)
    else:
        # Modo interativo
        menu_testes()