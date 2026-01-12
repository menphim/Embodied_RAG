# Embodied RAG アーキテクチャ図

このドキュメントでは、Embodied RAGシステムの構造をシーケンス図とアーキテクチャ図で説明します。

## 1. 全体アーキテクチャ図

```mermaid
graph TB
    subgraph "Entry Points"
        EXP[experiment.py]
        GSF[generate_semantic_forest.py]
        VIZ[graph_visualizer.py]
    end

    subgraph "Core Module (embodied_nav/)"
        ERAG[EmbodiedRAG<br/>メインオーケストレーター]
        ERET[EmbodiedRetriever<br/>検索ロジック]
        SRE[SpatialRelationshipExtractor<br/>階層グラフ構築]
        LLM[LLMInterface<br/>LLM通信]
        LLMH[LLMHierarchicalRetriever<br/>階層的検索]
        CFG[Config<br/>設定管理]
    end

    subgraph "Data Collection"
        ASE[AirSimExplorer<br/>テレオペ]
        OSE[OnlineSemanticExplorer<br/>リアルタイム]
        DSL[DirectSceneLogger<br/>全オブジェクト]
        ASU[AirSimUtils<br/>ドローン制御]
    end

    subgraph "External Services"
        OR[OpenRouter API]
        OAI[OpenAI API]
        AS[AirSim Simulator]
    end

    subgraph "Data Storage"
        GML[(GML Graph Files)]
        EMB[(Embedding Cache)]
    end

    EXP --> ERAG
    GSF --> SRE
    VIZ --> GML

    ERAG --> ERET
    ERAG --> LLM
    ERAG --> CFG
    ERET --> LLMH
    ERET --> LLM
    SRE --> LLM

    ASE --> ASU
    OSE --> ASE
    OSE --> SRE
    DSL --> ASU
    ASU --> AS

    LLM --> OR
    LLM --> OAI
    ERAG --> GML
    ERAG --> EMB
    SRE --> GML
```

## 2. オフラインワークフロー シーケンス図

```mermaid
sequenceDiagram
    participant User
    participant ASE as AirSimExplorer
    participant SRE as SpatialRelationshipExtractor
    participant LLM as LLMInterface
    participant GML as GML File
    participant ERAG as EmbodiedRAG
    participant RET as EmbodiedRetriever

    Note over User,RET: フェーズ1: データ収集
    User->>ASE: 探索開始
    ASE->>ASE: オブジェクト検出
    ASE->>GML: topological_graph.gml保存

    Note over User,RET: フェーズ2: Semantic Forest生成
    User->>SRE: generate_semantic_forest.py実行
    SRE->>GML: グラフ読み込み
    loop 各階層レベル
        SRE->>SRE: Agglomerative Clustering
        SRE->>SRE: クラスタノード作成
        SRE->>SRE: part_of エッジ追加
    end
    SRE->>SRE: 空間関係エッジ追加
    loop 各クラスタ
        SRE->>LLM: サマリー生成要求
        LLM-->>SRE: クラスタ名・説明
    end
    SRE->>GML: enhanced_semantic_graph.gml保存

    Note over User,RET: フェーズ3: クエリ処理
    User->>ERAG: experiment.py実行
    ERAG->>GML: グラフ読み込み
    ERAG->>ERAG: Embedding生成/キャッシュ確認
    ERAG->>RET: Retriever初期化
    User->>ERAG: クエリ送信
    ERAG->>RET: retrieve()
    RET-->>ERAG: 関連ノード
    ERAG->>LLM: ナビゲーション応答生成
    LLM-->>ERAG: 応答
    ERAG-->>User: 結果表示
```

## 3. クエリ処理 詳細シーケンス図

```mermaid
sequenceDiagram
    participant User
    participant ERAG as EmbodiedRAG
    participant EMB as Embedding生成
    participant RET as EmbodiedRetriever
    participant LLM as LLMInterface
    participant Graph as NetworkXグラフ

    User->>ERAG: query(query_text, query_type)

    ERAG->>EMB: クエリのEmbedding生成
    EMB->>LLM: text-embedding-ada-002
    LLM-->>EMB: 1536次元ベクトル
    EMB-->>ERAG: query_embedding

    alt method == SEMANTIC
        ERAG->>RET: _semantic_based_retrieval()
        RET->>Graph: 全ノードのEmbedding取得
        RET->>RET: コサイン類似度計算
        RET->>RET: 閾値フィルタリング (0.6)
        RET->>RET: Hierarchical Boost (1.2x)
        RET->>RET: Spatial Boost (+neighbor_avg*0.2)
        RET->>RET: Min-Max正規化
        RET-->>ERAG: top-k ノード
    else method == LLM_HIERARCHICAL
        ERAG->>RET: LLMHierarchicalRetriever
        loop 各階層レベル (top→bottom)
            RET->>Graph: 現レベルのノード取得
            RET->>LLM: select_best_node()
            LLM-->>RET: 最適ノード選択
            RET->>RET: 子ノードへ移動
        end
        RET-->>ERAG: 階層チェーン
    end

    ERAG->>RET: _build_context(nodes)
    RET-->>ERAG: コンテキスト文字列

    ERAG->>LLM: generate_navigation_response()
    LLM-->>ERAG: ナビゲーション応答

    ERAG-->>User: 応答 + ターゲットオブジェクト
```

## 4. Semantic Retrieval アルゴリズム図

```mermaid
flowchart TD
    Q[クエリ] --> E[Embedding生成]
    E --> S[コサイン類似度計算]

    subgraph "Initial Matching"
        S --> F[閾値フィルタ 0.6]
        F --> I[初期候補ノード]
    end

    subgraph "Hierarchical Boost"
        I --> H1{親ノードあり?}
        H1 -->|Yes| H2[親のサマリーと比較]
        H2 --> H3{関連性高い?}
        H3 -->|Yes| H4[スコア × 1.2]
        H3 -->|No| H5[スコア維持]
        H1 -->|No| H5
        H4 --> HO[階層ブースト後スコア]
        H5 --> HO
    end

    subgraph "Spatial Boost"
        HO --> SP1[近傍ノード検索]
        SP1 --> SP2[近傍の平均スコア計算]
        SP2 --> SP3[スコア += avg × 0.2]
        SP3 --> SPO[空間ブースト後スコア]
    end

    subgraph "Final Selection"
        SPO --> N[Min-Max正規化]
        N --> T[Top-k選択 k=5]
        T --> R[結果ノード]
    end
```

## 5. グラフ構造図

```mermaid
graph TB
    subgraph "Level 2 (Area)"
        A2[office_area<br/>summary: オフィスエリア...]
    end

    subgraph "Level 1 (Cluster)"
        C1[desk_cluster<br/>summary: デスク周辺...]
        C2[meeting_cluster<br/>summary: ミーティングスペース...]
    end

    subgraph "Level 0 (Objects)"
        O1[desk_1]
        O2[chair_1]
        O3[monitor_1]
        O4[table_1]
        O5[chair_2]
        O6[whiteboard_1]
    end

    O1 -->|part_of| C1
    O2 -->|part_of| C1
    O3 -->|part_of| C1
    O4 -->|part_of| C2
    O5 -->|part_of| C2
    O6 -->|part_of| C2

    C1 -->|part_of| A2
    C2 -->|part_of| A2

    O1 -.->|east, 1.5m| O2
    O2 -.->|north, 2.0m| O3
    O4 -.->|west, 3.0m| O5
    C1 -.->|south, 5.0m| C2
```

## 6. クラス関係図

```mermaid
classDiagram
    class EmbodiedRAG {
        -_graph_cache: Dict
        -retriever: EmbodiedRetriever
        -llm: LLMInterface
        +load_graph_to_rag(file)
        +query(text, type, position)
        +embedding_func(texts)
    }

    class EmbodiedRetriever {
        -graph: NetworkX
        -llm: LLMInterface
        -method: RetrievalMethod
        +retrieve(query, query_type, top_k)
        -_semantic_based_retrieval()
        -_apply_hierarchical_boost()
        -_apply_spatial_boost()
        -_build_context()
    }

    class LLMHierarchicalRetriever {
        -graph: NetworkX
        -llm: LLMInterface
        +retrieve(query, start_position)
        +select_best_node(query, nodes)
    }

    class SpatialRelationshipExtractor {
        -llm: LLMInterface
        +extract_relationships(objects)
        -_add_positional_relationships()
        -_get_cardinal_direction()
    }

    class LLMInterface {
        -provider: str
        -model: str
        +generate_response(prompt)
        +generate_community_summary(objects)
        +generate_navigation_response()
        +select_best_node()
    }

    class Config {
        <<dataclass>>
        +PATHS: PathConfig
        +SPATIAL: SpatialConfig
        +RETRIEVAL: RetrievalConfig
        +LLM: LLMConfig
    }

    EmbodiedRAG --> EmbodiedRetriever
    EmbodiedRAG --> LLMInterface
    EmbodiedRAG --> Config
    EmbodiedRetriever --> LLMHierarchicalRetriever
    EmbodiedRetriever --> LLMInterface
    SpatialRelationshipExtractor --> LLMInterface
```

## 7. オンラインワークフロー シーケンス図

```mermaid
sequenceDiagram
    participant User
    participant OSE as OnlineSemanticExplorer
    participant ASE as AirSimExplorer
    participant SRE as SpatialRelationshipExtractor
    participant LLM as LLMInterface
    participant Graph as グラフ

    User->>OSE: explore()開始

    loop 探索中
        OSE->>ASE: オブジェクト検出
        ASE-->>OSE: 検出オブジェクト
        OSE->>Graph: ノード追加

        alt N秒経過
            Note over OSE,LLM: 定期更新
            OSE->>Graph: 全オブジェクト抽出
            OSE->>SRE: extract_relationships()
            SRE->>SRE: クラスタリング
            SRE->>LLM: サマリー生成
            LLM-->>SRE: サマリー
            SRE-->>OSE: 更新済みSemantic Forest
        end
    end

    User->>OSE: 終了シグナル
    Note over OSE,LLM: 最終更新
    OSE->>SRE: 包括的更新
    SRE-->>OSE: 最終Semantic Forest
    OSE->>Graph: enhanced_graph.gml保存
```

## 主要コンポーネント説明

| コンポーネント | ファイル | 役割 |
|--------------|---------|------|
| **EmbodiedRAG** | `embodied_rag.py` | メインオーケストレーター、グラフ読み込み、Embedding管理 |
| **EmbodiedRetriever** | `embodied_retriever.py` | 2つの検索方式(SEMANTIC/LLM_HIERARCHICAL)を提供 |
| **SpatialRelationshipExtractor** | `spatial_relationship_extractor.py` | 階層的クラスタリングとグラフ構築 |
| **LLMInterface** | `llm.py` | OpenRouter/OpenAI/vLLM対応のLLM通信 |
| **Config** | `config.py` | 全設定パラメータの一元管理 |

## 検索方式の比較

| 方式 | 速度 | 特徴 |
|-----|------|------|
| **SEMANTIC** | ~0.7秒 | コサイン類似度 + 階層/空間ブースティング |
| **LLM_HIERARCHICAL** | ~10秒 | LLMによる階層的ノード選択 |
