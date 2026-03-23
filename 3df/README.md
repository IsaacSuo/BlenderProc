# 3df

独立的 3D-FRONT 批量渲染入口。

## 配置

编辑 [render.yaml](/home/fangsuo/py/BlenderProc/3df/render.yaml)：

- `paths.front_json_dir`: 3D-FRONT 的 `.json` 目录
- `paths.future_model_dir`: 3D-FUTURE-model 目录
- `paths.front_texture_dir`: 3D-FRONT-texture 目录
- `paths.output_dir`: 输出根目录

## 运行

批量渲染：

```bash
blenderproc run 3df/batch_render.py 3df/render.yaml
```

只渲染一个场景：

```bash
blenderproc run 3df/batch_render.py 3df/render.yaml --scene 00000000-0000-0000-0000-000000000000.json
```

限制场景数量：

```bash
blenderproc run 3df/batch_render.py 3df/render.yaml --limit 10
```

## 输出

每个场景会写到单独子目录：

- `0.hdf5`
- `scene_metadata.json`

批次根目录还会生成：

- `batch_summary.json`
