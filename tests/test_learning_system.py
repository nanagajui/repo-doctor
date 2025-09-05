#!/usr/bin/env python3
"""Test script for the Repo Doctor Learning System."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

from repo_doctor.learning import (
    FeatureExtractor,
    MLKnowledgeBase,
    MLDataStorage,
    DataQualityValidator,
    StrategySuccessPredictor,
    DependencyConflictPredictor,
    PatternDiscoveryEngine,
    AdaptiveLearningSystem,
    EnhancedResolutionAgent,
    EnhancedAnalysisAgent,
    LearningDashboard
)
from repo_doctor.models.analysis import Analysis, RepositoryInfo, DependencyInfo
from repo_doctor.models.resolution import Resolution, Strategy, StrategyType, ValidationResult, ValidationStatus
from repo_doctor.models.system import SystemProfile, HardwareInfo, SoftwareStack, GPUInfo


def create_sample_analysis() -> Analysis:
    """Create a sample analysis for testing."""
    # Create sample repository info
    repo_info = RepositoryInfo(
        owner="huggingface",
        name="transformers",
        url="https://github.com/huggingface/transformers",
        language="Python",
        size=1000000,
        star_count=50000,
        fork_count=12000,
        topics=["machine-learning", "nlp", "transformers", "pytorch"],
        has_dockerfile=True,
        has_conda_env=False,
        has_tests=True,
        has_ci=True,
        description="State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow"
    )
    
    # Create sample dependencies
    dependencies = [
        DependencyInfo(
            name="torch",
            version=">=1.9.0",
            gpu_required=True,
            source="requirements.txt"
        ),
        DependencyInfo(
            name="transformers",
            version=">=4.20.0",
            gpu_required=False,
            source="setup.py"
        ),
        DependencyInfo(
            name="numpy",
            version=">=1.19.0",
            gpu_required=False,
            source="requirements.txt"
        ),
        DependencyInfo(
            name="datasets",
            version=">=2.0.0",
            gpu_required=False,
            source="requirements.txt"
        )
    ]
    
    # Create sample analysis
    analysis = Analysis(
        repository=repo_info,
        dependencies=dependencies,
        python_version_required="3.8",
        cuda_version_required="11.8",
        min_memory_gb=8,
        min_gpu_memory_gb=4,
        compatibility_issues=[],
        confidence_score=0.85,
        analysis_time=2.5
    )
    
    return analysis


def create_sample_system_profile() -> SystemProfile:
    """Create a sample system profile for testing."""
    # Create hardware info
    hardware = HardwareInfo(
        cpu_cores=8,
        memory_gb=16.0,
        architecture="x86_64",
        gpus=[
            GPUInfo(
                name="NVIDIA RTX 3080",
                memory_gb=10.0,
                cuda_version="11.8"
            )
        ]
    )
    
    # Create software info
    software = SoftwareStack(
        python_version="3.9.0",
        pip_version="21.0.1",
        conda_version="4.10.3",
        docker_version="20.10.7",
        git_version="2.32.0",
        cuda_version="11.8"
    )
    
    # Create system profile
    profile = SystemProfile(
        hardware=hardware,
        software=software,
        container_runtime="docker",
        compute_score=0.9
    )
    
    return profile


def create_sample_resolution() -> Resolution:
    """Create a sample resolution for testing."""
    # Create strategy
    strategy = Strategy(
        type=StrategyType.DOCKER,
        requirements={
            "estimated_setup_time": 300,
            "requires_gpu": True,
            "requires_cuda": True
        }
    )
    
    # Create generated files
    generated_files = [
        {
            "name": "Dockerfile",
            "content": "FROM nvidia/cuda:11.8-devel-ubuntu20.04\n...",
            "path": "./Dockerfile"
        },
        {
            "name": "docker-compose.yml",
            "content": "version: '3.8'\nservices:\n  app:\n...",
            "path": "./docker-compose.yml"
        }
    ]
    
    # Create resolution
    resolution = Resolution(
        strategy=strategy,
        generated_files=generated_files,
        setup_commands=["pip install -r requirements.txt"],
        instructions="## Setup Instructions\n\n1. Build the Docker image\n2. Run with GPU support\n...",
        estimated_size_mb=2048,
        confidence_score=0.8
    )
    
    return resolution


def create_sample_validation_result() -> ValidationResult:
    """Create a sample validation result for testing."""
    return ValidationResult(
        status=ValidationStatus.SUCCESS,
        duration=45.2,
        logs=["Build successful", "Tests passed", "GPU detected"],
        error_message=None
    )


async def test_feature_extraction():
    """Test feature extraction functionality."""
    print("🧪 Testing Feature Extraction...")
    
    # Create sample data
    analysis = create_sample_analysis()
    system_profile = create_sample_system_profile()
    resolution = create_sample_resolution()
    validation_result = create_sample_validation_result()
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    repo_features = extractor.extract_repository_features(analysis)
    system_features = extractor.extract_system_features(system_profile)
    resolution_features = extractor.extract_resolution_features(resolution)
    learning_features = extractor.extract_learning_features(analysis, resolution, validation_result)
    
    print(f"✅ Repository features: {len(repo_features)} features extracted")
    print(f"✅ System features: {len(system_features)} features extracted")
    print(f"✅ Resolution features: {len(resolution_features)} features extracted")
    print(f"✅ Learning features: {len(learning_features)} features extracted")
    
    # Print some key features
    print(f"   - ML dependencies: {repo_features.get('ml_dependencies', 0)}")
    print(f"   - GPU dependencies: {repo_features.get('gpu_dependencies', 0)}")
    print(f"   - GPU count: {system_features.get('gpu_count', 0)}")
    print(f"   - Strategy type: {resolution_features.get('strategy_type', 'unknown')}")
    print(f"   - Success: {learning_features.get('success', False)}")
    
    return {
        "repo_features": repo_features,
        "system_features": system_features,
        "resolution_features": resolution_features,
        "learning_features": learning_features
    }


async def test_ml_knowledge_base():
    """Test ML knowledge base functionality."""
    print("\n🧪 Testing ML Knowledge Base...")
    
    # Create temporary storage path
    storage_path = Path("/tmp/repo_doctor_test_learning")
    storage_path.mkdir(exist_ok=True)
    
    try:
        # Initialize ML knowledge base
        ml_kb = MLKnowledgeBase(storage_path)
        
        # Create sample data
        analysis = create_sample_analysis()
        system_profile = create_sample_system_profile()
        resolution = create_sample_resolution()
        validation_result = create_sample_validation_result()
        
        # Record ML analysis
        commit_hash = ml_kb.record_ml_analysis(analysis, resolution, validation_result, system_profile)
        print(f"✅ Recorded ML analysis with commit hash: {commit_hash}")
        
        # Get ML recommendations
        recommendations = ml_kb.get_ml_recommendations(analysis, system_profile)
        print(f"✅ Generated ML recommendations: {len(recommendations.get('recommendations', []))} recommendations")
        
        # Get learning insights
        insights = ml_kb.get_learning_insights(analysis)
        print(f"✅ Generated learning insights: {len(insights)} insights")
        
        # Get learning metrics
        metrics = ml_kb.get_learning_metrics()
        print(f"✅ Learning metrics: {metrics}")
        
        return {
            "commit_hash": commit_hash,
            "recommendations": recommendations,
            "insights": insights,
            "metrics": metrics
        }
        
    finally:
        # Cleanup
        import shutil
        if storage_path.exists():
            shutil.rmtree(storage_path)


async def test_strategy_predictor():
    """Test strategy prediction functionality."""
    print("\n🧪 Testing Strategy Predictor...")
    
    # Create temporary storage path
    storage_path = Path("/tmp/repo_doctor_test_predictor")
    storage_path.mkdir(exist_ok=True)
    
    try:
        # Initialize ML data storage
        ml_storage = MLDataStorage(storage_path)
        
        # Create sample training data
        training_data = []
        for i in range(20):  # Create 20 sample records
            analysis = create_sample_analysis()
            system_profile = create_sample_system_profile()
            resolution = create_sample_resolution()
            
            # Create training record
            record = {
                "timestamp": time.time(),
                "repository_features": {
                    "ml_dependencies": 2 + (i % 3),
                    "gpu_dependencies": 1 if i % 2 == 0 else 0,
                    "total_dependencies": 10 + i,
                    "is_ml_repo": True,
                    "requires_gpu": i % 2 == 0
                },
                "system_features": {
                    "gpu_count": 1 if i % 2 == 0 else 0,
                    "memory_gb": 16 + (i % 4) * 8,
                    "cpu_cores": 8 + (i % 4) * 2
                },
                "resolution_features": {
                    "strategy_type": ["docker", "conda", "venv"][i % 3]
                },
                "outcome": {
                    "success": i % 4 != 0,  # 75% success rate
                    "duration": 30 + i * 2
                }
            }
            training_data.append(record)
            ml_storage.store_training_record(record)
        
        # Initialize strategy predictor
        predictor = StrategySuccessPredictor(storage=ml_storage)
        
        # Train model
        print("   Training strategy predictor...")
        train_result = predictor.train()
        print(f"✅ Training completed: {train_result}")
        
        # Test predictions
        repo_features = {
            "ml_dependencies": 3,
            "gpu_dependencies": 1,
            "total_dependencies": 15,
            "is_ml_repo": True,
            "requires_gpu": True
        }
        system_features = {
            "gpu_count": 1,
            "memory_gb": 32,
            "cpu_cores": 16
        }
        
        # Get strategy recommendations
        recommendations = predictor.get_strategy_recommendations(repo_features, system_features)
        print(f"✅ Strategy recommendations: {len(recommendations)} recommendations")
        
        for rec in recommendations:
            print(f"   - {rec['strategy']}: {rec['success_probability']:.2f} probability ({rec['confidence']} confidence)")
        
        return {
            "train_result": train_result,
            "recommendations": recommendations
        }
        
    finally:
        # Cleanup
        import shutil
        if storage_path.exists():
            shutil.rmtree(storage_path)


async def test_pattern_discovery():
    """Test pattern discovery functionality."""
    print("\n🧪 Testing Pattern Discovery...")
    
    # Create temporary storage path
    storage_path = Path("/tmp/repo_doctor_test_patterns")
    storage_path.mkdir(exist_ok=True)
    
    try:
        # Initialize ML knowledge base
        ml_kb = MLKnowledgeBase(storage_path)
        pattern_engine = PatternDiscoveryEngine(ml_kb)
        
        # Create sample training data
        training_data = []
        for i in range(30):  # Create 30 sample records
            analysis = create_sample_analysis()
            system_profile = create_sample_system_profile()
            resolution = create_sample_resolution()
            validation_result = create_sample_validation_result()
            
            # Record analysis
            ml_kb.record_ml_analysis(analysis, resolution, validation_result, system_profile)
        
        # Discover patterns
        print("   Discovering patterns...")
        patterns = pattern_engine.discover_patterns(min_support=0.1)
        print(f"✅ Discovered {len(patterns)} patterns")
        
        for pattern in patterns[:3]:  # Show top 3 patterns
            print(f"   - {pattern.pattern_id}: {pattern.description} (confidence: {pattern.confidence:.2f})")
        
        # Get pattern summary
        summary = pattern_engine.get_pattern_summary()
        print(f"✅ Pattern summary: {summary}")
        
        return {
            "patterns": patterns,
            "summary": summary
        }
        
    finally:
        # Cleanup
        import shutil
        if storage_path.exists():
            shutil.rmtree(storage_path)


async def test_learning_dashboard():
    """Test learning dashboard functionality."""
    print("\n🧪 Testing Learning Dashboard...")
    
    # Create temporary storage path
    storage_path = Path("/tmp/repo_doctor_test_dashboard")
    storage_path.mkdir(exist_ok=True)
    
    try:
        # Initialize ML knowledge base
        ml_kb = MLKnowledgeBase(storage_path)
        dashboard = LearningDashboard(ml_kb)
        
        # Create sample data
        for i in range(15):  # Create 15 sample records
            analysis = create_sample_analysis()
            system_profile = create_sample_system_profile()
            resolution = create_sample_resolution()
            validation_result = create_sample_validation_result()
            
            # Record analysis
            ml_kb.record_ml_analysis(analysis, resolution, validation_result, system_profile)
        
        # Get dashboard metrics
        metrics = dashboard.get_dashboard_metrics()
        print(f"✅ Dashboard metrics: {metrics}")
        
        # Get learning insights
        insights = dashboard.get_learning_insights(5)
        print(f"✅ Learning insights: {len(insights)} insights")
        
        # Get top patterns
        patterns = dashboard.get_top_patterns(3)
        print(f"✅ Top patterns: {len(patterns)} patterns")
        
        # Get learning status
        status = dashboard.get_learning_status_summary()
        print(f"✅ Learning status: {status}")
        
        # Get recommendations
        recommendations = dashboard.get_learning_recommendations()
        print(f"✅ Learning recommendations: {len(recommendations)} recommendations")
        
        return {
            "metrics": metrics,
            "insights": insights,
            "patterns": patterns,
            "status": status,
            "recommendations": recommendations
        }
        
    finally:
        # Cleanup
        import shutil
        if storage_path.exists():
            shutil.rmtree(storage_path)


async def test_enhanced_agents():
    """Test enhanced agents functionality."""
    print("\n🧪 Testing Enhanced Agents...")
    
    # Create temporary storage path
    storage_path = Path("/tmp/repo_doctor_test_agents")
    storage_path.mkdir(exist_ok=True)
    
    try:
        # Initialize enhanced agents
        analysis_agent = EnhancedAnalysisAgent(str(storage_path))
        resolution_agent = EnhancedResolutionAgent(str(storage_path))
        
        # Test analysis agent
        print("   Testing enhanced analysis agent...")
        analysis = await analysis_agent.analyze("https://github.com/huggingface/transformers")
        print(f"✅ Analysis completed with confidence: {analysis.confidence_score}")
        
        # Test ML insights
        insights = analysis_agent.get_ml_insights(analysis)
        print(f"✅ ML insights: {len(insights)} insights")
        
        # Test strategy recommendations
        recommendations = analysis_agent.get_ml_recommendations(analysis)
        print(f"✅ ML recommendations: {len(recommendations.get('recommendations', []))} recommendations")
        
        # Test resolution agent
        print("   Testing enhanced resolution agent...")
        resolution = await resolution_agent.resolve(analysis)
        print(f"✅ Resolution completed with strategy: {resolution.strategy.type.value}")
        
        # Test learning metrics
        metrics = resolution_agent.get_learning_metrics()
        print(f"✅ Learning metrics: {metrics}")
        
        return {
            "analysis": analysis,
            "resolution": resolution,
            "insights": insights,
            "recommendations": recommendations,
            "metrics": metrics
        }
        
    finally:
        # Cleanup
        import shutil
        if storage_path.exists():
            shutil.rmtree(storage_path)


async def main():
    """Run all learning system tests."""
    print("🚀 Starting Repo Doctor Learning System Tests\n")
    
    try:
        # Test feature extraction
        feature_results = await test_feature_extraction()
        
        # Test ML knowledge base
        kb_results = await test_ml_knowledge_base()
        
        # Test strategy predictor
        predictor_results = await test_strategy_predictor()
        
        # Test pattern discovery
        pattern_results = await test_pattern_discovery()
        
        # Test learning dashboard
        dashboard_results = await test_learning_dashboard()
        
        # Test enhanced agents
        agent_results = await test_enhanced_agents()
        
        print("\n🎉 All Learning System Tests Completed Successfully!")
        print("\n📊 Test Summary:")
        print(f"   - Feature extraction: ✅ {len(feature_results['repo_features'])} features")
        print(f"   - ML knowledge base: ✅ {kb_results['commit_hash']}")
        print(f"   - Strategy predictor: ✅ {len(predictor_results['recommendations'])} recommendations")
        print(f"   - Pattern discovery: ✅ {len(pattern_results['patterns'])} patterns")
        print(f"   - Learning dashboard: ✅ {dashboard_results['status']['health_status']} health")
        print(f"   - Enhanced agents: ✅ {len(agent_results['insights'])} insights")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
