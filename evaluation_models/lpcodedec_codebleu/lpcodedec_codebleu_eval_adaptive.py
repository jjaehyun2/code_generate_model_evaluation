class AdaptiveHybridEvaluator(HybridCodeEvaluator):
    """적응적 가중치를 사용하는 고급 하이브리드 평가기"""
    
    def __init__(self):
        super().__init__()
        
    def evaluate_with_adaptive_strategy(self, 
                                      reference_code: str,
                                      generated_codes: List[str],
                                      instruction: str = "",
                                      lang: str = "python") -> Dict[str, Any]:
        """코드 특성에 따른 적응적 가중치 적용 평가"""
        
        # 기본 평가 수행
        base_results = []
        for gen_code in generated_codes:
            base_result = self.evaluate_single_pair(reference_code, gen_code, lang)
            base_results.append(base_result)
        
        # 코드 특성 분석
        code_characteristics = self._analyze_code_characteristics(reference_code, generated_codes)
        
        # 적응적 가중치 계산
        adaptive_weights = self._calculate_adaptive_weights(code_characteristics)
        
        # 다중 전략 평가
        strategy_results = {}
        
        # 전략 1: 균등 가중치
        strategy_results['balanced'] = self._evaluate_with_weights(
            base_results, {'style': 0.5, 'semantic': 0.5}
        )
        
        # 전략 2: 스타일 중심 (LPcodedec 논문의 주요 발견 반영)
        strategy_results['style_focused'] = self._evaluate_with_weights(
            base_results, {'style': 0.7, 'semantic': 0.3}
        )
        
        # 전략 3: 의미 중심 (CodeBLEU 강조)
        strategy_results['semantic_focused'] = self._evaluate_with_weights(
            base_results, {'style': 0.3, 'semantic': 0.7}
        )
        
        # 전략 4: 적응적 가중치
        strategy_results['adaptive'] = self._evaluate_with_weights(
            base_results, adaptive_weights
        )
        
        # 앙상블 점수 계산
        ensemble_result = self._calculate_ensemble_score(strategy_results)
        
        return {
            'instruction': instruction,
            'reference_code': reference_code,
            'strategy_results': strategy_results,
            'adaptive_weights': adaptive_weights,
            'ensemble_result': ensemble_result,
            'code_characteristics': code_characteristics,
            'recommended_strategy': self._recommend_strategy(strategy_results, code_characteristics)
        }
    
    def _analyze_code_characteristics(self, reference_code: str, generated_codes: List[str]) -> Dict[str, Any]:
        """코드 특성 분석"""
        ref_features = self.lpcodedec_analyzer.extract_lpcodedec_features(reference_code)
        
        # 코드 복잡도 분석
        ref_complexity = self._calculate_complexity(reference_code)
        gen_complexities = [self._calculate_complexity(code) for code in generated_codes]
        
        # 코드 길이 분석
        ref_length = len(reference_code.split('\n'))
        gen_lengths = [len(code.split('\n')) for code in generated_codes]
        
        # LPcodedec 특징별 변이성 분석
        gen_features = [self.lpcodedec_analyzer.extract_lpcodedec_features(code) for code in generated_codes]
        feature_variations = np.std(gen_features, axis=0) if gen_features else np.zeros(10)
        
        return {
            'ref_complexity': ref_complexity,
            'avg_gen_complexity': np.mean(gen_complexities),
            'complexity_ratio': np.mean(gen_complexities) / max(ref_complexity, 1),
            'ref_length': ref_length,
            'avg_gen_length': np.mean(gen_lengths),
            'length_ratio': np.mean(gen_lengths) / max(ref_length, 1),
            'feature_variations': feature_variations.tolist(),
            'high_variation_features': np.where(feature_variations > np.mean(feature_variations))[0].tolist()
        }
    
    def _calculate_complexity(self, code: str) -> float:
        """순환 복잡도 기반 코드 복잡도 계산"""
        complexity = 1
        
        # 제어 구조
        complexity += code.count('if ')
        complexity += code.count('elif ')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('try:')
        complexity += code.count('except')
        complexity += code.count('and ')
        complexity += code.count('or ')
        complexity += code.count('class ')
        complexity += code.count('def ')
        
        return complexity
    
    def _calculate_adaptive_weights(self, characteristics: Dict[str, Any]) -> Dict[str, float]:
        """코드 특성 기반 적응적 가중치 계산"""
        
        complexity_ratio = characteristics['complexity_ratio']
        length_ratio = characteristics['length_ratio']
        high_variation_features = characteristics['high_variation_features']
        
        # 기본 가중치
        style_weight = 0.5
        semantic_weight = 0.5
        
        # 복잡도 기반 조정
        if complexity_ratio > 1.3:  # 생성 코드가 더 복잡
            style_weight += 0.1  # 스타일 차이 더 중요
            semantic_weight -= 0.1
        elif complexity_ratio < 0.8:  # 생성 코드가 더 단순
            style_weight -= 0.1
            semantic_weight += 0.1  # 의미 보존이 더 중요
        
        # 길이 기반 조정
        if length_ratio > 1.2:  # 생성 코드가 더 김
            semantic_weight += 0.05  # 의미적 정확성 중요
            style_weight -= 0.05
        
        # LPcodedec 특징 변이성 기반 조정
        if len(high_variation_features) > 5:  # 스타일 변이가 큰 경우
            style_weight += 0.1
            semantic_weight -= 0.1
        
        # 정규화
        total = style_weight + semantic_weight
        return {
            'style': style_weight / total,
            'semantic': semantic_weight / total
        }
    
    def _evaluate_with_weights(self, base_results: List[Dict], weights: Dict[str, float]) -> Dict[str, Any]:
        """주어진 가중치로 재평가"""
        weighted_results = []
        
        for result in base_results:
            weighted_score = (
                weights['style'] * result['style_similarity'] +
                weights['semantic'] * result['semantic_similarity']
            )
            
            weighted_result = result.copy()
            weighted_result['weighted_score'] = weighted_score
            weighted_results.append(weighted_result)
        
        best_result = max(weighted_results, key=lambda x: x['weighted_score'])
        
        return {
            'best_result': best_result,
            'all_results': weighted_results,
            'weights_used': weights
        }
    
    def _calculate_ensemble_score(self, strategy_results: Dict) -> Dict[str, Any]:
        """앙상블 점수 계산"""
        strategy_scores = {}
        for strategy, result in strategy_results.items():
            strategy_scores[strategy] = result['best_result']['weighted_score']
        
        # 논문 기반 전략별 가중치 (LPcodedec이 우수한 성능을 보였으므로 스타일 중심 전략에 더 가중치)
        strategy_weights = {
            'balanced': 0.25,
            'style_focused': 0.35,  # LPcodedec 논문 결과 반영
            'semantic_focused': 0.25,
            'adaptive': 0.15
        }
        
        ensemble_score = sum(
            strategy_weights[strategy] * score 
            for strategy, score in strategy_scores.items()
        )
        
        return {
            'ensemble_score': ensemble_score,
            'strategy_scores': strategy_scores,
            'strategy_weights': strategy_weights
        }
    
    def _recommend_strategy(self, strategy_results: Dict, characteristics: Dict) -> str:
        """코드 특성 기반 최적 전략 권장"""
        
        # LPcodedec 논문의 주요 발견 반영
        complexity_ratio = characteristics['complexity_ratio']
        high_variation_features = characteristics['high_variation_features']
        
        # Comment Ratio가 논문에서 가장 중요한 특징으로 나타남
        if 7 in high_variation_features:  # Comment Ratio는 인덱스 7
            return 'style_focused'
        
        # 복잡도 차이가 클 때
        if abs(complexity_ratio - 1.0) > 0.3:
            return 'adaptive'
        
        # 기본적으로는 균형잡힌 전략
        return 'balanced'

# 사용 예시 (버전 2)
def main_v2():
    MODEL_PATH = "./finetuned_model"
    code_generator = ModelCodeGenerator(MODEL_PATH)
    evaluator = AdaptiveHybridEvaluator()
    
    with open("./test_dataset.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    all_results = []
    
    for idx, sample in enumerate(test_data[:3]):  # 3개 샘플 테스트
        instruction = sample["instruction"]
        reference_code = sample["output"]
        
        print(f"\n[{idx+1}/3] 적응적 평가 중...")
        print(f"Instruction: {instruction[:80]}...")
        
        # 코드 생성
        generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
        
        # 적응적 평가 수행
        result = evaluator.evaluate_with_adaptive_strategy(
            reference_code, generated_codes, instruction, lang="python"
        )
        
        # 결과 출력
        print(f"권장 전략: {result['recommended_strategy']}")
        print(f"앙상블 점수: {result['ensemble_result']['ensemble_score']:.4f}")
        print(f"적응적 가중치: 스타일={result['adaptive_weights']['style']:.3f}, "
              f"의미={result['adaptive_weights']['semantic']:.3f}")
        
        # 전략별 최고 성능 비교
        print("전략별 최고 점수:")
        for strategy, strategy_result in result['strategy_results'].items():
            best_score = strategy_result['best_result']['weighted_score']
            best_idx = strategy_result['best_result']['generation_index']
            print(f"  {strategy}: {best_score:.4f} (후보 {best_idx})")
        
        all_results.append(result)
    
    # 결과 저장
    with open("adaptive_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # main()      # 버전 1 실행
    main_v2()     # 버전 2 실행
