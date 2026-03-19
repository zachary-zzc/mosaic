"""
构建最小示例图并保存，供 run_example 使用。
不依赖 mosaic 的 LLM/embedding，仅生成图与 tags 的磁盘文件。
运行方式（在 LongtermMemory 根目录）:
  python example/build_minimal_graph.py
"""
from __future__ import annotations

import json
import os
import pickle


class _MinimalClassNode:
    """最小类节点，与 mosaic ClassNode 属性兼容，便于 pickle 后被主代码加载。"""
    class_id = None
    class_name = ""
    attributes = set()
    operations = set()
    unclassified = set()
    _instances = []


def build_minimal_graph():
    """构建一个仅含一个类和两个实例的图（networkx.Graph 兼容结构）。"""
    try:
        from networkx import Graph
    except ImportError:
        raise SystemExit("请安装 networkx: pip install networkx")

    graph = Graph()
    node = _MinimalClassNode()
    node.class_id = "class_1"
    node.class_name = "BookClub"
    node.attributes = {"book_name", "joined_time", "favorite_character"}
    node.operations = {"join", "read"}
    node.unclassified = set()
    node._instances = [
        {
            "instance_id": "instance_1",
            "instance_name": "User A book club",
            "attributes": {
                "book_name": {"description": "当前在读的书", "value": "The Great Gatsby", "occurred": None, "recorded_at": "1 Jan 2024"},
                "joined_time": {"description": "加入时间", "value": "last week", "occurred": None, "recorded_at": "1 Jan 2024"},
                "favorite_character": {"description": "喜欢的角色", "value": "Nick", "occurred": None, "recorded_at": "1 Jan 2024"},
            },
            "operations": {"read": {"description": "阅读书籍"}},
            "uninstance_field": "We are reading The Great Gatsby. I love it. I like Nick. He is the narrator.",
            "message_labels": [1, 3, 5],
        },
    ]
    graph.add_node(node)
    return graph


def main():
    example_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(example_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    graph = build_minimal_graph()
    graph_path = os.path.join(out_dir, "graph_small.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"已保存图: {graph_path}")

    tags = [
        {"class_id": "class_1", "instance_id": "instance_1", "keywords": ["book", "Great Gatsby", "Nick", "book club", "reading"]}
    ]
    tags_path = os.path.join(out_dir, "tags_small.json")
    with open(tags_path, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)
    print(f"已保存 tags: {tags_path}")
    return graph_path, tags_path


if __name__ == "__main__":
    main()
