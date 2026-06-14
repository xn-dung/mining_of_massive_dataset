```mermaid
graph TD
    subgraph DVC_Storage [1. DVC & Storage]
        A[(Nguồn dữ liệu\nTLC Web + Context data Holidays/Weather)] -->|Tải Parquet định kỳ| B(Google Drive\nRemote Storage)
        B -->|dvc add| C{Data Versioning\nDVC Hash}
    end

    subgraph Data_Processing [2. Data Processing]
        C --> D[EDA\nPhân tích xu hướng & Phân phối]
        D --> E[Preprocessing\nLàm sạch & Xây dựng các trường dữ liệu cần thiết]
        E --> E1[Data Preparing\ncho STDN]
        E --> E2[Data Preparing\ncho DMVST-net]
    end

    subgraph Training_MLflow [3. Training & MLflow Tracking]
        E1 --> F(Huấn luyện:\nSTDN Model)
        E2 --> G(Huấn luyện:\nDMVST-net Model)
        
        F -.->|Log Params, Metrics| H[(MLflow Server)]
        G -.->|Log Params, Metrics| H
        
        H --> I{So sánh & Chọn\nBest Model}
    end

    subgraph Monitoring_Trigger [4. Monitoring & Auto-Finetune]
        I --> J{Kiểm tra MAE\n> Ngưỡng?}
        J -->|Không vượt ngưỡng| K[Lưu trữ Model\nSẵn sàng Phục vụ]
        J -->|Vượt ngưỡng cảnh báo| L[Trigger:\nKích hoạt Finetune]
        L -.->|Tự động điều phối qua Airflow| B
    end

    %% Định dạng màu sắc chuẩn bằng mã Hex (không dùng var CSS)
    classDef storage fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;
    classDef process fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000;
    classDef training fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef trigger fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000;

    class A,B,C storage;
    class D,E,E1,E2 process;
    class F,G,H,I training;
    class J,K,L trigger;