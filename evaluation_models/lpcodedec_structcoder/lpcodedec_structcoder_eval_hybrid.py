#lpdedoc_structcode_eval_v1.py
import ast
import re
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

class LPcodedecAnalyzer:
    """LPcodedec 논문의 코딩 스타일 특징 분석기
    
    논문의 핵심 발견:
    - Comment Ratio가 모든 언어에서 가장 중요한 특징 (F-statistic 최고)
    - Python에서는 Readability 특징이 가장 효과적
    - C++/Java에서는 Code Structure 특징이 가장 효과적
    """
    
    def __init__(self):
        self.naming_patterns = {
            'camelCase': re.compile(r'^[a-z][a-zA-Z0-9]*$'),
            'PascalCase': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
            'snake_case': re.compile(r'^[a-z_][a-z0-9_]*$'),
            'UPPER_SNAKE_CASE': re.compile(r'^[A-Z_][A-Z0-9_]*$'),
        }
        
        # 논문의 Table 3 결과를 반영한 특징별 중요도
        self.feature_importance = {
            'comment_ratio': 1.0,        # 가장 중요 (모든 언어에서 최고 F-statistic)
            'function_length': 0.8,      # 두 번째로 중요
            'variable_naming': 0.7,      # Java에서 중요
            'class_naming': 0.6,         # C++에서 중요
            'indentation_consistency': 0.5,
            'nesting_depth': 0.5,
            'function_naming': 0.4,
            'constant_naming': 0.4,
            'function_name_length': 0.4,
            'variable_name_length': 0.3
        }
    
    def extract_lpcodedec_features(self, code: str) -> np.ndarray:
        """LPcodedec 논문의 10가지 스타일 특징 추출
        
        특징 순서 (논문 Table 8 기준):
        1-4: Naming Consistency (함수, 변수, 클래스, 상수)
        5-7: Code Structure (들여쓰기, 함수길이, 중첩깊이)  
        8-10: Readability (주석비율, 함수명길이, 변수명길이)
        """
        try:
            tree = ast.parse(code)
        except:
            return np.zeros(10, dtype=np.float32)
        
        # 1-4. Naming Consistency 특징들
        naming_features = self._analyze_naming_consistency(tree)
        
        # 5-7. Code Structure 특징들  
        structure_features = self._analyze_code_structure(code, tree)
        
        # 8-10. Readability 특징들 (Comment Ratio 포함)
        readability_features = self._analyze_readability(code, tree)
        
        features = [
            naming_features['function_naming'],      # 1
            naming_features['variable_naming'],      # 2  
            naming_features['class_naming'],         # 3
            naming_features['constant_naming'],      # 4
            structure_features['indentation_consistency'], # 5
            structure_features['avg_function_length'],     # 6
            structure_features['avg_nesting_depth'],       # 7
            readability_features['comment_ratio'],         # 8 (가장 중요!)
            readability_features['avg_function_name_length'], # 9
            readability_features['avg_variable_name_length']  # 10
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _analyze_naming_consistency(self, tree: ast.AST) -> Dict[str, float]:
        """네이밍 일관성 분석 (논문 Section 4.1.1)"""
        functions, variables, classes, constants = [], [], [], []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Name):
                if node.id.isupper() and len(node.id) > 1:
                    constants.append(node.id)
                else:
                    variables.append(node.id)
        
        return {
            'function_naming': self._get_naming_consistency(functions),
            'variable_naming': self._get_naming_consistency(variables),
            'class_naming': self._get_naming_consistency(classes),
            'constant_naming': self._get_naming_consistency(constants)
        }
    
    def _get_naming_consistency(self, names: List[str]) -> float:
        """가장 일관된 네이밍 패턴의 비율 계산"""
        if not names:
            return 0.0
        
        pattern_counts = {}
        for name in names:
            matched = False
            for pattern_name, pattern in self.naming_patterns.items():
                if pattern.match(name):
                    pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
                    matched = True
                    break
            
            if not matched:
                pattern_counts['other'] = pattern_counts.get('other', 0) + 1
        
        return max(pattern_counts.values()) / len(names) if pattern_counts else 0.0
    
    def _analyze_code_structure(self, code: str, tree: ast.AST) -> Dict[str, float]:
        """코드 구조 분석 (논문 Section 4.1.2)"""
        lines = code.split('\n')
        indentations = []
        function_lengths = []
        nesting_depths = []
        
        # 들여쓰기 패턴 분석
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indentations.append(indent)
        
        # 함수별 분석
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno + 1 if node.end_lineno else 1
                function_lengths.append(func_lines)
                nesting_depths.append(self._calculate_nesting_depth(node))
        
        # 들여쓰기 일관성 계산
        indent_consistency = 0.0
        if indentations:
            indent_counts = Counter(indentations)
            most_common_indent = indent_counts.most_common(1)[0][0]
            indent_consistency = indent_counts[most_common_indent] / len(indentations)
        
        return {
            'indentation_consistency': indent_consistency,
            'avg_function_length': np.mean(function_lengths) if function_lengths else 0,
            'avg_nesting_depth': np.mean(nesting_depths) if nesting_depths else 0
        }
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """중첩 깊이 계산 (제어 구조의 복잡도)"""
        max_depth = 0
        
        def calculate_depth(n, current_depth=0):
            nonlocal max_depth
            if isinstance(n, (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.ExceptHandler)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(n):
                calculate_depth(child, current_depth)
        
        calculate_depth(node)
        return max_depth
    
    def _analyze_readability(self, code: str, tree: ast.AST) -> Dict[str, float]:
        """가독성 분석 (논문 Section 4.1.3) - Comment Ratio가 핵심!"""
        lines = code.split('\n')
        total_lines = len([line for line in lines if line.strip()])  # 빈 줄 제외
        
        # 주석 라인 계산 (논문에서 가장 중요한 특징)
        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') or '"""' in stripped or "'''" in stripped:
                comment_lines += 1
        
        # 함수명과 변수명 길이 분석
        function_name_lengths = []
        variable_name_lengths = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name_lengths.append(len(node.name))
            elif isinstance(node, ast.Name):
                variable_name_lengths.append(len(node.id))
        
        return {
            'comment_ratio': comment_lines / total_lines if total_lines > 0 else 0,
            'avg_function_name_length': np.mean(function_name_lengths) if function_name_lengths else 0,
            'avg_variable_name_length': np.mean(variable_name_lengths) if variable_name_lengths else 0
        }

class StructCoderAnalyzer:
    """StructCoder 방식의 구조적 특징 분석기
    
    StructCoder의 핵심 아이디어:
    - AST와 데이터 플로우 그래프를 동시 예측하는 구조 인식 트랜스포머
    - 단순한 토큰 시퀀스를 넘어서 코드의 구조적 의미 이해
    - 변수 의존성, 제어 흐름, 함수 호출 관계 등을 종합 분석
    """
    
    def __init__(self):
        self.ast_node_types = [
            'Module', 'FunctionDef', 'ClassDef', 'Return', 'Delete', 'Assign',
            'AugAssign', 'AnnAssign', 'For', 'While', 'If', 'With', 'Raise',
            'Try', 'Assert', 'Import', 'ImportFrom', 'Global', 'Nonlocal',
            'Expr', 'Pass', 'Break', 'Continue', 'Call', 'Compare', 'BinOp',
            'UnaryOp', 'Lambda', 'IfExp', 'Dict', 'Set', 'ListComp', 'SetComp',
            'DictComp', 'GeneratorExp', 'Await', 'Yield', 'YieldFrom'
        ]
        
    def extract_structural_features(self, code: str) -> Dict[str, Any]:
        """StructCoder 방식의 종합적 구조 분석"""
        try:
            tree = ast.parse(code)
        except:
            return self._get_empty_features()
        
        # 1. AST 구조 특징 (노드 분포, 깊이, 복잡도)
        ast_features = self._extract_ast_structural_features(tree)
        
        # 2. 데이터 플로우 그래프 특징
        dfg_features = self._extract_dataflow_features(tree)
        
        # 3. 제어 흐름 특징  
        cfg_features = self._extract_control_flow_features(tree)
        
        # 4. 함수 호출 그래프 특징
        call_graph_features = self._extract_call_graph_features(tree)
        
        # 5. 변수 의존성 특징
        dependency_features = self._extract_dependency_features(tree)
        
        return {
            'ast_features': ast_features,
            'dfg_features': dfg_features,
            'cfg_features': cfg_features,
            'call_graph_features': call_graph_features,
            'dependency_features': dependency_features,
            'combined_structural_vector': self._combine_features(
                ast_features, dfg_features, cfg_features, 
                call_graph_features, dependency_features
            )
        }
    
    def _extract_ast_structural_features(self, tree: ast.AST) -> np.ndarray:
        """AST 구조적 특징 추출 (StructCoder의 핵심)"""
        node_counts = defaultdict(int)
        total_nodes = 0
        max_depth = 0
        leaf_nodes = 0
        branching_factors = []
        
        def analyze_node(node, depth=0):
            nonlocal max_depth, total_nodes, leaf_nodes
            
            node_type = type(node).__name__
            node_counts[node_type] += 1
            total_nodes += 1
            max_depth = max(max_depth, depth)
            
            children = list(ast.iter_child_nodes(node))
            if not children:
                leaf_nodes += 1
            else:
                branching_factors.append(len(children))
            
            for child in children:
                analyze_node(child, depth + 1)
        
        analyze_node(tree)
        
        # AST 구조적 메트릭 계산
        features = []
        
        # 1. 주요 노드 타입별 정규화된 빈도
        for node_type in self.ast_node_types:
            frequency = node_counts[node_type] / total_nodes if total_nodes > 0 else 0
            features.append(frequency)
        
        # 2. 구조적 복잡도 메트릭
        features.extend([
            max_depth,                                                    # 최대 깊이
            leaf_nodes / total_nodes if total_nodes > 0 else 0,         # 리프 노드 비율
            np.mean(branching_factors) if branching_factors else 0,      # 평균 분기 계수
            np.std(branching_factors) if branching_factors else 0,       # 분기 계수 표준편차
            len(set(node_counts.keys())),                                # 고유 노드 타입 수
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_dataflow_features(self, tree: ast.AST) -> np.ndarray:
        """데이터 플로우 그래프 특징 (StructCoder의 핵심 아이디어)"""
        variables = set()
        assignments = []
        usages = []
        def_use_chains = []
        
        class DataFlowVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_scope_vars = set()
                
            def visit_Name(self, node):
                variables.add(node.id)
                if isinstance(node.ctx, ast.Store):
                    assignments.append(node.id)
                    self.current_scope_vars.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    usages.append(node.id)
                    if node.id in self.current_scope_vars:
                        def_use_chains.append((node.id, 'local'))
                    else:
                        def_use_chains.append((node.id, 'non_local'))
                
                self.generic_visit(node)
        
        visitor = DataFlowVisitor()
        visitor.visit(tree)
        
        # 데이터 플로우 메트릭
        num_vars = len(variables)
        assignment_counts = Counter(assignments)
        usage_counts = Counter(usages)
        
        features = [
            num_vars,                                                    # 총 변수 수
            len(assignments),                                            # 총 할당 수
            len(usages),                                                 # 총 사용 수
            len(assignments) / num_vars if num_vars > 0 else 0,         # 할당 밀도
            len(usages) / num_vars if num_vars > 0 else 0,              # 사용 밀도
            len(assignment_counts) / num_vars if num_vars > 0 else 0,    # 할당 변수 비율
            len(usage_counts) / num_vars if num_vars > 0 else 0,        # 사용 변수 비율
            np.mean(list(assignment_counts.values())) if assignment_counts else 0,  # 평균 할당 빈도
            np.mean(list(usage_counts.values())) if usage_counts else 0,           # 평균 사용 빈도
            len([chain for chain in def_use_chains if chain[1] == 'local']) / len(def_use_chains) if def_use_chains else 0,  # 지역 변수 비율
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_control_flow_features(self, tree: ast.AST) -> np.ndarray:
        """제어 흐름 특징 추출"""
        control_structures = {
            'if_count': 0, 'for_count': 0, 'while_count': 0,
            'try_count': 0, 'with_count': 0, 'nested_loops': 0
        }
        
        function_complexities = []
        
        class ControlFlowVisitor(ast.NodeVisitor):
            def __init__(self):
                self.loop_depth = 0
                
            def visit_If(self, node):
                control_structures['if_count'] += 1
                self.generic_visit(node)
                
            def visit_For(self, node):
                control_structures['for_count'] += 1
                self.loop_depth += 1
                if self.loop_depth > 1:
                    control_structures['nested_loops'] += 1
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_While(self, node):
                control_structures['while_count'] += 1
                self.loop_depth += 1
                if self.loop_depth > 1:
                    control_structures['nested_loops'] += 1
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_Try(self, node):
                control_structures['try_count'] += 1
                self.generic_visit(node)
                
            def visit_With(self, node):
                control_structures['with_count'] += 1
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                # 함수별 순환 복잡도 계산
                complexity = 1  # 기본 복잡도
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                        complexity += 1
                function_complexities.append(complexity)
                self.generic_visit(node)
        
        visitor = ControlFlowVisitor()
        visitor.visit(tree)
        
        total_control_structures = sum(control_structures.values())
        
        features = [
            control_structures['if_count'],
            control_structures['for_count'],
            control_structures['while_count'],
            control_structures['try_count'],
            control_structures['with_count'],
            control_structures['nested_loops'],
            total_control_structures,
            np.mean(function_complexities) if function_complexities else 0,
            np.std(function_complexities) if function_complexities else 0,
            max(function_complexities) if function_complexities else 0,
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_call_graph_features(self, tree: ast.AST) -> np.ndarray:
        """함수 호출 그래프 특징"""
        function_calls = []
        function_definitions = []
        builtin_calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    function_calls.append(func_name)
                    # 내장 함수 확인
                    if func_name in ['len', 'print', 'range', 'int', 'str', 'list', 'dict', 'set']:
                        builtin_calls.append(func_name)
            elif isinstance(node, ast.FunctionDef):
                function_definitions.append(node.name)
        
        call_counts = Counter(function_calls)
        builtin_counts = Counter(builtin_calls)
        
        features = [
            len(function_calls),                                         # 총 함수 호출 수
            len(function_definitions),                                   # 정의된 함수 수
            len(builtin_calls),                                         # 내장 함수 호출 수
            len(set(function_calls)),                                   # 고유 함수 호출 수
            len(builtin_calls) / len(function_calls) if function_calls else 0,  # 내장 함수 비율
            np.mean(list(call_counts.values())) if call_counts else 0,  # 평균 호출 빈도
            max(call_counts.values()) if call_counts else 0,           # 최대 호출 빈도
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_dependency_features(self, tree: ast.AST) -> np.ndarray:
        """변수 의존성 특징"""
        dependencies = []
        scopes = []
        
        class DependencyVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_scope = set()
                self.scope_stack = []
                
            def visit_FunctionDef(self, node):
                # 새로운 스코프 시작
                self.scope_stack.append(self.current_scope.copy())
                scopes.append(len(self.current_scope))
                
                # 함수 매개변수 추가
                for arg in node.args.args:
                    self.current_scope.add(arg.arg)
                
                self.generic_visit(node)
                
                # 스코프 복원
                self.current_scope = self.scope_stack.pop() if self.scope_stack else set()
                
            def visit_Assign(self, node):
                # 할당문에서 의존성 분석
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.current_scope.add(target.id)
                
                # 우변의 변수 사용 확인
                for var_node in ast.walk(node.value):
                    if isinstance(var_node, ast.Name) and isinstance(var_node.ctx, ast.Load):
                        if var_node.id in self.current_scope:
                            dependencies.append(('local', var_node.id))
                        else:
                            dependencies.append(('global', var_node.id))
                
                self.generic_visit(node)
        
        visitor = DependencyVisitor()
        visitor.visit(tree)
        
        local_deps = len([d for d in dependencies if d[0] == 'local'])
        global_deps = len([d for d in dependencies if d == 'global'])
        
        features = [
            len(dependencies),                                          # 총 의존성 수
            local_deps,                                                # 지역 의존성 수
            global_deps,                                               # 전역 의존성 수
            local_deps / len(dependencies) if dependencies else 0,     # 지역 의존성 비율
            len(scopes),                                               # 스코프 수
            np.mean(scopes) if scopes else 0,                         # 평균 스코프 크기
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _combine_features(self, ast_features, dfg_features, cfg_features, 
                         call_graph_features, dependency_features) -> np.ndarray:
        """모든 구조적 특징을 결합한 통합 벡터"""
        return np.concatenate([
            ast_features,
            dfg_features, 
            cfg_features,
            call_graph_features,
            dependency_features
        ])
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """파싱 실패시 빈 특징 반환"""
        return {
            'ast_features': np.zeros(len(self.ast_node_types) + 5, dtype=np.float32),
            'dfg_features': np.zeros(10, dtype=np.float32),
            'cfg_features': np.zeros(10, dtype=np.float32),
            'call_graph_features': np.zeros(7, dtype=np.float32),
            'dependency_features': np.zeros(6, dtype=np.float32),
            'combined_structural_vector': np.zeros(len(self.ast_node_types) + 38, dtype=np.float32)
        }

class LPStructHybridEvaluator:
    """LPcodedec + StructCoder 순수 하이브리드 평가기
    
    두 방식의 조합 전략:
    1. LPcodedec: 빠르고 효율적인 스타일 분석 (1,343배 빠름)
    2. StructCoder: 정확하고 심층적인 구조 분석
    
    CodeBLEU 제외 - 스타일과 구조적 측면만 평가
    """
    
    def __init__(self, 
                 style_weight: float = 0.5,      # LPcodedec 스타일 분석
                 structural_weight: float = 0.5): # StructCoder 구조 분석
        
        self.lpcodedec_analyzer = LPcodedecAnalyzer()
        self.structcoder_analyzer = StructCoderAnalyzer()
        
        # 가중치 정규화
        total_weight = style_weight + structural_weight
        self.style_weight = style_weight / total_weight
        self.structural_weight = structural_weight / total_weight
        
        print(f"StructCoder + LPcodedec 하이브리드 가중치:")
        print(f"  - 스타일 (LPcodedec): {self.style_weight:.3f}")
        print(f"  - 구조 (StructCoder): {self.structural_weight:.3f}")
    
    def evaluate_single_pair(self, reference_code: str, generated_code: str) -> Dict[str, Any]:
        """단일 코드 쌍 2차원 하이브리드 평가 (스타일 + 구조)"""
        
        # 1. LPcodedec 스타일 유사도
        ref_style_features = self.lpcodedec_analyzer.extract_lpcodedec_features(reference_code)
        gen_style_features = self.lpcodedec_analyzer.extract_lpcodedec_features(generated_code)
        style_similarity = self._cosine_similarity(ref_style_features, gen_style_features)
        
        # 2. StructCoder 구조적 유사도
        structural_similarity, structural_details = self._calculate_structural_similarity(
            reference_code, generated_code
        )
        
        # 3. 2차원 하이브리드 점수 계산 (스타일 + 구조만)
        hybrid_score = (
            self.style_weight * style_similarity +
            self.structural_weight * structural_similarity
        )
        
        return {
            'hybrid_score': hybrid_score,
            'style_similarity': style_similarity,
            'structural_similarity': structural_similarity,
            'style_features_ref': ref_style_features.tolist(),
            'style_features_gen': gen_style_features.tolist(),
            'structural_details': structural_details,
            'weights': {
                'style_weight': self.style_weight,
                'structural_weight': self.structural_weight
            }
        }
    
    def _calculate_structural_similarity(self, ref_code: str, gen_code: str) -> Tuple[float, Dict]:
        """StructCoder 방식의 구조적 유사도 계산"""
        ref_struct = self.structcoder_analyzer.extract_structural_features(ref_code)
        gen_struct = self.structcoder_analyzer.extract_structural_features(gen_code)
        
        # 각 구조적 측면별 유사도 계산
        similarities = {}
        
        # 1. AST 구조 유사도 (가장 중요 - 40% 가중치)
        ast_similarity = self._cosine_similarity(
            ref_struct['ast_features'], gen_struct['ast_features']
        )
        similarities['ast_similarity'] = ast_similarity
        
        # 2. 데이터 플로우 유사도 (StructCoder의 핵심 - 30% 가중치)
        dfg_similarity = self._cosine_similarity(
            ref_struct['dfg_features'], gen_struct['dfg_features']
        )
        similarities['dfg_similarity'] = dfg_similarity
        
        # 3. 제어 흐름 유사도 (20% 가중치)
        cfg_similarity = self._cosine_similarity(
            ref_struct['cfg_features'], gen_struct['cfg_features']
        )
        similarities['cfg_similarity'] = cfg_similarity
        
        # 4. 함수 호출 그래프 유사도 (7% 가중치)
        call_similarity = self._cosine_similarity(
            ref_struct['call_graph_features'], gen_struct['call_graph_features']
        )
        similarities['call_similarity'] = call_similarity
        
        # 5. 변수 의존성 유사도 (3% 가중치)
        dep_similarity = self._cosine_similarity(
            ref_struct['dependency_features'], gen_struct['dependency_features']
        )
        similarities['dependency_similarity'] = dep_similarity
        
        # 가중 평균으로 최종 구조적 유사도 계산
        structural_similarity = (
            0.40 * ast_similarity +
            0.30 * dfg_similarity +
            0.20 * cfg_similarity +
            0.07 * call_similarity +
            0.03 * dep_similarity
        )
        
        return structural_similarity, similarities
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def evaluate_best_of_n(self, 
                          reference_code: str,
                          generated_codes: List[str],
                          instruction: str = "") -> Dict[str, Any]:
        """Best-of-N 방식 2차원 하이브리드 평가 (스타일 + 구조)"""
        
        if not generated_codes:
            raise ValueError("생성된 코드 목록이 비어있습니다.")
        
        evaluation_results = []
        print(f"  {len(generated_codes)}개 후보 코드 평가 중...")
        
        for i, gen_code in enumerate(generated_codes):
            try:
                result = self.evaluate_single_pair(reference_code, gen_code)
                result['generation_index'] = i
                result['generated_code'] = gen_code
                evaluation_results.append(result)
                
                print(f"    후보 {i}: 하이브리드={result['hybrid_score']:.4f} "
                      f"(스타일={result['style_similarity']:.3f}, "
                      f"구조={result['structural_similarity']:.3f})")
                
            except Exception as e:
                print(f"생성 코드 {i} 평가 오류: {e}")
                evaluation_results.append({
                    'hybrid_score': 0.0,
                    'style_similarity': 0.0,
                    'structural_similarity': 0.0,
                    'generation_index': i,
                    'generated_code': gen_code,
                    'error': str(e)
                })
        
        # 최고 점수 선택
        best_result = max(evaluation_results, key=lambda x: x['hybrid_score'])
        
        # 통계 계산
        valid_results = [r for r in evaluation_results if 'error' not in r]
        scores = [r['hybrid_score'] for r in valid_results]
        style_scores = [r['style_similarity'] for r in valid_results]
        structural_scores = [r['structural_similarity'] for r in valid_results]
        
        return {
            'instruction': instruction,
            'reference_code': reference_code,
            'best_result': best_result,
            'all_results': evaluation_results,
            'n_generations': len(generated_codes),
            'statistics': {
                'hybrid_scores': {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                },
                'component_scores': {
                    'style': {'mean': np.mean(style_scores), 'best_idx': np.argmax(style_scores)},
                    'structural': {'mean': np.mean(structural_scores), 'best_idx': np.argmax(structural_scores)}
                }
            },
            'diversity_metrics': {
                'score_diversity': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
                'best_improvement': best_result['hybrid_score'] - np.mean(scores),
                'style_structural_correlation': np.corrcoef(style_scores, structural_scores)[0,1] if len(style_scores) > 1 else 0
            }
        }

class ModelCodeGenerator:
    """파인튜닝된 모델을 사용한 코드 생성기"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"코드 생성 모델을 {self.device}에서 로드 중...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        print("모델 로드 완료!")
    
    def generate_codes(self, instruction: str, num_candidates: int = 3, 
                      max_new_tokens: int = 512) -> List[str]:
        """주어진 코드 설명으로 N개의 서로 다른 코드 생성"""
        
        # 프롬프트 구성 (Instruction 형식)
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            generation_outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,          # 적당한 창의성
                top_p=0.95,              # 품질 유지
                num_return_sequences=num_candidates,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3   # 반복 방지
            )
        
        # 생성된 코드 후처리
        generated_codes = []
        for output in generation_outputs:
            # 입력 부분 제거하고 생성된 부분만 추출
            generated_text = self.tokenizer.decode(
                output[len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            
            # 코드 정제 (주석, 공백 정리)
            clean_code = self._clean_generated_code(generated_text)
            generated_codes.append(clean_code)
        
        return generated_codes
    
    def _clean_generated_code(self, code: str) -> str:
        """생성된 코드 정제"""
        lines = code.split('\n')
        clean_lines = []
        
        for line in lines:
            # 빈 줄 스킵
            if not line.strip():
                continue
            # 특수 토큰 제거 (### 또는 백틱 3개로 시작하는 마크다운 코드 블록)
            if line.strip().startswith('###') or line.strip().startswith('```'):
                continue
            clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
# 메인 함수
def main():
    """StructCoder + LPcodedec 하이브리드 평가 실행"""
    
    print("=== StructCoder + LPcodedec 순수 하이브리드 평가 시스템 ===")
    print("논문 기반 스타일 분석 (LPcodedec) + 구조 인식 분석 (StructCoder)")
    print("※ CodeBLEU 제외 - 스타일과 구조적 측면만 평가")
    print()
    
    # 1. 모델 및 평가기 초기화
    #MODEL_PATH = "jack0503/code_generate_explain"  # 파인튜닝된 모델 경로
    #MODEL_PATH = "jack0503/code-usage-model"
    #MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
    MODEL_PATH = "./finetuned_model/finetuned_V1_optimized"

    try:
        code_generator = ModelCodeGenerator(MODEL_PATH)
        evaluator = LPStructHybridEvaluator(
            style_weight=0.5,      # LPcodedec 스타일 분석
            structural_weight=0.5  # StructCoder 구조 분석
        )
        print()
        
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 2. 테스트 데이터셋 로드
    try:
        with open("./dataset/test_data.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
        print(f"테스트 데이터셋 로드 완료: {len(test_data)}개 샘플")
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return
    
    all_case_results = []
    num_test_samples = min(5, len(test_data))  # 처음 5개 샘플만 테스트
    
    print(f"\n총 {num_test_samples}개 테스트 케이스 평가 시작...\n")
    
    for idx, sample in enumerate(test_data[:num_test_samples]):
        instruction = sample["instruction"]
        reference_code = sample["output"]
        
        print(f"[{idx+1}/{num_test_samples}] 평가 중...")
        print(f"Instruction: {instruction[:100]}{'...' if len(instruction) > 100 else ''}")
        
        # 3. 코드 설명으로 3개 코드 생성
        try:
            generated_codes = code_generator.generate_codes(instruction, num_candidates=3)
            print(f"  코드 생성 완료: {len(generated_codes)}개 후보")
        except Exception as e:
            print(f"  코드 생성 실패: {e}")
            continue
        
        # 4. Best-of-3 하이브리드 평가 수행
        try:
            result = evaluator.evaluate_best_of_n(
                reference_code,
                generated_codes,
                instruction
            )
            
            # 5. 결과 저장
            case_result = {
                "case_index": idx,
                "instruction": instruction,
                "reference_code": reference_code,
                "best_candidate_index": result["best_result"]["generation_index"],
                "best_hybrid_score": result["best_result"]["hybrid_score"],
                "best_style_similarity": result["best_result"]["style_similarity"],
                "best_structural_similarity": result["best_result"]["structural_similarity"],
                "score_improvement": result["diversity_metrics"]["best_improvement"],
                "score_diversity": result["diversity_metrics"]["score_diversity"],
                "style_structural_correlation": result["diversity_metrics"]["style_structural_correlation"],
                "component_best_indices": result["statistics"]["component_scores"],
                "structural_details": result["best_result"]["structural_details"]
            }
            all_case_results.append(case_result)
            
            # 개별 결과 출력
            print(f"  ✓ 최고 하이브리드 점수: {result['best_result']['hybrid_score']:.4f}")
            print(f"  ✓ 선택된 최고 후보: {result['best_result']['generation_index']}번")
            print(f"  ✓ 점수 개선도: {result['diversity_metrics']['best_improvement']:.4f}")
            print(f"  ✓ 구성 요소별 점수:")
            print(f"    - LPcodedec 스타일: {result['best_result']['style_similarity']:.4f}")
            print(f"    - StructCoder 구조: {result['best_result']['structural_similarity']:.4f}")
            
            # StructCoder 세부 구조 분석 출력
            struct_details = result['best_result']['structural_details']
            print(f"  ✓ StructCoder 세부 분석:")
            print(f"    - AST 구조: {struct_details['ast_similarity']:.4f}")
            print(f"    - 데이터 플로우: {struct_details['dfg_similarity']:.4f}")
            print(f"    - 제어 흐름: {struct_details['cfg_similarity']:.4f}")
            print(f"    - 함수 호출: {struct_details['call_similarity']:.4f}")
            print(f"    - 변수 의존성: {struct_details['dependency_similarity']:.4f}")
            print()
            
        except Exception as e:
            print(f"  평가 실패: {e}")
            continue
    
    # 6. 전체 결과 저장
    with open("lpstruct_pure_hybrid_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy(all_case_results), f, ensure_ascii=False, indent=2)
    
    # 7. 전체 통계 분석 및 출력
    if all_case_results:
        print("=" * 60)
        print("전체 StructCoder + LPcodedec 순수 하이브리드 평가 결과")
        print("=" * 60)
        
        # 기본 통계
        avg_hybrid_score = np.mean([r["best_hybrid_score"] for r in all_case_results])
        avg_improvement = np.mean([r["score_improvement"] for r in all_case_results])
        avg_style = np.mean([r["best_style_similarity"] for r in all_case_results])
        avg_structural = np.mean([r["best_structural_similarity"] for r in all_case_results])
        avg_diversity = np.mean([r["score_diversity"] for r in all_case_results])
        avg_correlation = np.mean([r["style_structural_correlation"] for r in all_case_results])
        
        print(f"평균 최고 하이브리드 점수: {avg_hybrid_score:.4f}")
        print(f"평균 점수 개선도: {avg_improvement:.4f}")
        print(f"평균 점수 다양성: {avg_diversity:.4f}")
        print(f"스타일-구조 상관관계: {avg_correlation:.4f}")
        print()
        
        print("구성 요소별 평균 성능:")
        print(f"  - LPcodedec 스타일 유사도: {avg_style:.4f}")
        print(f"  - StructCoder 구조적 유사도: {avg_structural:.4f}")
        print()
        
        # 후보 선택 분포 분석
        selection_counts = Counter([r["best_candidate_index"] for r in all_case_results])
        print("Best-of-3 후보 선택 분포:")
        for idx in range(3):
            count = selection_counts.get(idx, 0)
            percentage = count / len(all_case_results) * 100 if all_case_results else 0
            print(f"  후보 {idx}: {count}회 ({percentage:.1f}%)")
        print()
        
        # StructCoder 세부 구조 분석 평균
        structural_components = ['ast_similarity', 'dfg_similarity', 'cfg_similarity', 'call_similarity', 'dependency_similarity']
        print("StructCoder 구조적 구성요소별 평균 점수:")
        for component in structural_components:
            avg_score = np.mean([r["structural_details"][component] for r in all_case_results])
            print(f"  - {component}: {avg_score:.4f}")
        
        print(f"\n결과가 'lpstruct_pure_hybrid_evaluation_results.json'에 저장되었습니다.")
        
    else:
        print("평가 가능한 결과가 없습니다.")

if __name__ == "__main__":
    main()
