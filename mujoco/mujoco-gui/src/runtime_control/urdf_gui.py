"""Local browser GUI for validated URDF to MJCF conversion."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import webbrowser

from .urdf_converter import convert_urdf


def _dialog(command):
    executable = shutil.which("zenity")
    if executable is None:
        raise RuntimeError("系统未安装 zenity，无法打开文件选择窗口")
    result = subprocess.run(
        [executable, *command], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 1:  # User cancelled the dialog.
        return None
    if result.returncode != 0:
        raise RuntimeError("文件选择窗口失败: %s" % result.stderr.strip())
    return result.stdout.strip() or None


def _open_directory(path):
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("xdg-open")
    if executable is None:
        raise RuntimeError("系统未安装 xdg-open，无法打开文件夹")
    subprocess.Popen(
        [executable, str(directory)], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return directory


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>URDF 转 MuJoCo MJCF</title>
<style>
:root{--bg:#07101d;--panel:#101c2d;--line:#2b4160;--text:#edf6ff;--muted:#91a8c5;--blue:#43a2ff;--cyan:#48ddd0}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#102541,var(--bg) 55%);color:var(--text);font:14px system-ui,sans-serif}
main{max-width:820px;margin:6vh auto;padding:26px;background:#101c2dee;border:1px solid var(--line);border-radius:16px;box-shadow:0 20px 70px #0008}
h1{margin:4px 0 6px;font-size:27px}h2{font-size:14px;color:var(--cyan);margin:24px 0 10px}.lead,small{color:var(--muted)}
label{display:block;margin:13px 0 5px;color:#b9cbe0}input,textarea,button{width:100%;padding:11px;border:1px solid var(--line);border-radius:8px;background:#15263b;color:var(--text)}
textarea{min-height:78px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.checks{display:flex;gap:24px;margin:18px 0}.checks label{display:flex;align-items:center;gap:7px;margin:0}.checks input{width:auto}
.pathrow{display:grid;grid-template-columns:1fr 130px;gap:8px}.pathrow button{margin:0}
button{cursor:pointer;font-weight:700}.primary{background:#14609e;border-color:#2789d7}.danger{margin-top:10px;border-color:#88404d}.primary:hover{background:#1974bd}
#status{margin-top:18px;padding:13px;white-space:pre-wrap;border-radius:8px;background:#091321;color:var(--muted)}.ok{color:#55e1a1!important;border:1px solid #338566}.bad{color:#ff7887!important;border:1px solid #a74755}
</style><main><small>MUJOCO RUNTIME CONTROL 2.1</small><h1>URDF → MuJoCo MJCF</h1>
<p class="lead">转换、编译验证，并自动打开原生 MuJoCo 查看模型效果。</p>
<h2>文件</h2><label>URDF 文件路径</label><div class="pathrow"><input id="urdf" placeholder="/absolute/path/robot.urdf"><button id="pickUrdf">选择 URDF…</button></div>
<label>输出 MJCF XML 路径</label><div class="pathrow"><input id="output" placeholder="/absolute/path/robot.xml"><button id="pickOutput">选择保存位置…</button></div>
<small id="defaultFolder"></small>
<label>Mesh 目录（可选）</label><input id="meshDir" placeholder="留空时相对 URDF 文件解析">
<label>ROS package 映射（可选，每行 NAME=/path）</label><textarea id="packages" placeholder="my_robot=/path/to/my_robot"></textarea>
<h2>转换参数</h2><div class="row"><div><label>机器人出生高度（m）</label><input id="spawn" type="number" value="0.45" step="0.01"></div><div><label>力矩倍率</label><input id="effort" type="number" value="1.0" step="0.1" min="0.01"></div></div>
<div class="checks"><label><input id="collision" type="checkbox">仅显示碰撞模型</label><label><input id="preview" type="checkbox" checked>转换后自动打开 MuJoCo</label></div>
<button id="convert" class="primary">转换并验证</button><button id="openOutput">打开转换结果文件夹</button><button id="exit" class="danger">退出转换器</button><div id="status">等待输入</div>
</main><script>
const get=id=>document.getElementById(id),status=get('status');
get('urdf').onchange=()=>{if(!get('output').value&&get('urdf').value.toLowerCase().endsWith('.urdf'))get('output').value=get('urdf').value.slice(0,-5)+'.xml'};
async function api(path,body=null){let options=body===null?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};let response=await fetch(path,options),data=await response.json();if(!response.ok)throw Error(data.error);return data}
get('pickUrdf').onclick=async()=>{try{let data=await api('/api/pick-urdf');if(!data.cancelled){get('urdf').value=data.path;get('output').value=data.suggested_output;get('meshDir').value=data.suggested_mesh_dir||''}}catch(e){alert(e.message)}};
get('pickOutput').onclick=async()=>{try{let data=await api('/api/pick-output',{urdf:get('urdf').value,output:get('output').value});if(!data.cancelled)get('output').value=data.path}catch(e){alert(e.message)}};
get('openOutput').onclick=async()=>{try{await api('/api/open-output',{output:get('output').value});}catch(e){alert(e.message)}};
api('/api/info').then(data=>{get('defaultFolder').textContent='默认输出位置：与选中的 URDF 文件同目录'}).catch(()=>{});
get('convert').onclick=async()=>{status.className='';status.textContent='正在转换并编译验证…';try{let packages={};get('packages').value.split(/\n/).map(x=>x.trim()).filter(Boolean).forEach(line=>{let i=line.indexOf('=');if(i<1)throw Error('package 映射格式应为 NAME=/path');packages[line.slice(0,i).trim()]=line.slice(i+1).trim()});let payload={urdf:get('urdf').value,output:get('output').value,mesh_dir:get('meshDir').value||null,package_dirs:packages,spawn_height:Number(get('spawn').value),effort_multiplier:Number(get('effort').value),collision_only:get('collision').checked,preview:get('preview').checked};let response=await fetch('/api/convert',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),data=await response.json();if(!response.ok)throw Error(data.error);status.className='ok';status.textContent='✓ 转换成功\n'+data.path+'\n模型维度：nq='+data.nq+' nv='+data.nv+' nu='+data.nu+'\nbody='+data.nbody+' geom='+data.ngeom+(data.preview?'\nMuJoCo viewer 已启动':'')}catch(e){status.className='bad';status.textContent='转换失败：\n'+e.message}};
get('exit').onclick=async()=>{if(!confirm('退出转换器并关闭由它启动的 MuJoCo 预览吗？'))return;try{await fetch('/api/shutdown',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});}finally{document.body.innerHTML='<main><h1>URDF 转换器已退出</h1><p>服务和 MuJoCo 预览已停止，可以关闭此页面。</p></main>'}};
</script></html>'''


class UrdfGuiServer:
    def __init__(self, host="127.0.0.1", port=8770, python_executable=sys.executable, output_dir=None):
        self.python_executable = str(python_executable)
        self.output_dir = Path(output_dir or "generated/converted_models").expanduser().resolve()
        self.preview_process = None
        self.thread = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, value, status=200, content_type="application/json"):
                body = value.encode("utf-8") if isinstance(value, str) else json.dumps(value, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type + "; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in {"/", "/index.html"}:
                    self.reply(PAGE, content_type="text/html")
                elif self.path == "/api/info":
                    self.reply({"output_dir": str(owner.output_dir)})
                elif self.path == "/api/pick-urdf":
                    try:
                        self.reply(owner.pick_urdf())
                    except Exception as error:
                        self.reply({"error": str(error)}, 400)
                else:
                    self.reply({"error": "not found"}, 404)

            def do_POST(self):
                if self.path not in {
                    "/api/convert", "/api/shutdown", "/api/pick-output",
                    "/api/open-output",
                }:
                    self.reply({"error": "not found"}, 404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 1_000_000:
                        raise ValueError("invalid request size")
                    payload = json.loads(self.rfile.read(length))
                    if self.path == "/api/shutdown":
                        self.reply({"shutdown": True})
                        threading.Thread(target=owner.request_shutdown, daemon=True).start()
                    elif self.path == "/api/pick-output":
                        self.reply(owner.pick_output(payload))
                    elif self.path == "/api/open-output":
                        self.reply(owner.open_output(payload))
                    else:
                        self.reply(owner.convert(payload))
                except Exception as error:
                    self.reply({"error": str(error)}, 400)

            def log_message(self, *_):
                pass

        try:
            self.server = ThreadingHTTPServer((host, int(port)), Handler)
        except OSError as error:
            if getattr(error, "errno", None) == 98:
                raise OSError("URDF 转换器端口 %s:%s 已被占用" % (host, port)) from error
            raise

    @property
    def url(self):
        host, port = self.server.server_address[:2]
        return "http://%s:%d/" % (host, port)

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
        return self

    def _stop_preview(self):
        if self.preview_process is not None and self.preview_process.poll() is None:
            self.preview_process.terminate()
            try:
                self.preview_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.preview_process.kill()

    def pick_urdf(self):
        path = _dialog([
            "--file-selection", "--title=选择 URDF 文件",
            "--file-filter=URDF files | *.urdf", "--file-filter=All files | *",
        ])
        if path is None:
            return {"cancelled": True}
        urdf = Path(path).resolve()
        mesh_candidates = (urdf.parent / "meshes", urdf.parent.parent / "meshes")
        mesh_dir = next((item for item in mesh_candidates if item.is_dir()), None)
        return {
            "cancelled": False, "path": str(urdf),
            "suggested_output": str(urdf.with_suffix(".xml")),
            "suggested_mesh_dir": "" if mesh_dir is None else str(mesh_dir),
        }

    def pick_output(self, payload):
        current = str(payload.get("output", "")).strip()
        urdf = str(payload.get("urdf", "")).strip()
        name = Path(urdf).stem + ".xml" if urdf else "robot.xml"
        initial = (
            Path(current).expanduser() if current else
            (Path(urdf).expanduser().with_suffix(".xml") if urdf else self.output_dir / name)
        )
        path = _dialog([
            "--file-selection", "--save", "--confirm-overwrite",
            "--title=选择 MJCF XML 保存位置", "--filename=%s" % initial,
            "--file-filter=MuJoCo XML | *.xml",
        ])
        return {"cancelled": path is None, "path": path}

    def open_output(self, payload):
        output = str(payload.get("output", "")).strip()
        directory = Path(output).expanduser().resolve().parent if output else self.output_dir
        return {"opened": str(_open_directory(directory))}

    def convert(self, payload):
        urdf = str(payload.get("urdf", "")).strip()
        output = str(payload.get("output", "")).strip()
        if not urdf or not output:
            raise ValueError("URDF 和输出 XML 路径不能为空")
        result = convert_urdf(
            urdf, output,
            package_dirs=payload.get("package_dirs") or {},
            mesh_dir=payload.get("mesh_dir") or None,
            spawn_height=float(payload.get("spawn_height", 0.45)),
            effort_multiplier=float(payload.get("effort_multiplier", 1.0)),
            collision_only=bool(payload.get("collision_only", False)),
            validate=True,
        )
        if payload.get("preview", True):
            self._stop_preview()
            self.preview_process = subprocess.Popen([
                self.python_executable, "-m", "runtime_control.mjcf_viewer", result["path"]
            ], cwd=str(Path(result["path"]).parent))
            result["preview"] = True
        else:
            result["preview"] = False
        return result

    def request_shutdown(self):
        self._stop_preview()
        self.server.shutdown()

    def close(self):
        self._stop_preview()
        if self.thread is not None:
            self.server.shutdown()
            self.thread.join(timeout=2)
            self.thread = None
        self.server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Browser GUI for URDF to MJCF conversion")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    server = UrdfGuiServer(args.host, args.port, args.python_executable).start()
    print("URDF converter: %s" % server.url)
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
