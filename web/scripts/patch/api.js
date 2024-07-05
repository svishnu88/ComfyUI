import { app } from "../app.js";
class ComfyApi extends EventTarget {
  #registered = new Set();
  getUserConfig = async () => ({ storage: "browser", migrated: true });

  constructor() {
    super();
    this.api_host = location.host;
    this.api_base = location.pathname.split("/").slice(0, -1).join("/");
    this.initialClientId = sessionStorage.getItem("clientId");
  }

  apiURL(route) {
    return this.api_base + route;
  }

  fetchApi(route, options) {
    if (!options) {
      options = {};
    }
    if (!options.headers) {
      options.headers = {};
    }
    options.headers["Comfy-User"] = this.user;
    return fetch(this.apiURL(route), options);
  }

  addEventListener(type, callback, options) {
    super.addEventListener(type, callback, options);
    this.#registered.add(type);
  }

  async getUserData(file, options) {
    return this.fetchApi(`/userdata/${encodeURIComponent(file)}`, options);
  }

  async listUserData(dir, recurse, split) {
    const resp = await this.fetchApi(
      `/userdata?${new URLSearchParams({
        recurse,
        dir,
        split,
      })}`
    );
    if (resp.status === 404) return [];
    if (resp.status !== 200) {
      throw new Error(
        `Error getting user data list '${dir}': ${resp.status} ${resp.statusText}`
      );
    }
    return resp.json();
  }

  async getSystemStats() {}

  async getSettings() {
    return (await this.fetchApi("/settings")).json();
  }

  init() {
    console.log("init");

    window.addEventListener(
      "message",
      (event) => {
        if (event.data === "run") {
          console.log("Run command received from parent");
          performRunAction()
            .then((data) => {
              event.source.postMessage(
                {
                  type: "result",
                  data: data,
                },
                event.origin
              );
            })
            .catch((error) => {
              event.source.postMessage(
                {
                  type: "error",
                  message: error.message,
                },
                event.origin
              );

              console.log("err", error);
            });
        }
      },
      false
    );
    function performRunAction() {
      return new Promise((resolve, reject) => {
        app
          .graphToPrompt()
          .then((p) => {
            const json = JSON.stringify(p.output, null, 2);
            console.log(json);
            resolve({ message: json });
          })
          .catch((error) => {
            console.log("cat", error);
            reject(error);
          });
      });
    }
  }

  async getExtensions() {
    const resp = await this.fetchApi("/extensions", { cache: "no-store" });
    return await resp.json();
  }

  async getNodeDefs() {
    const resp = await this.fetchApi("/object_info", { cache: "no-store" });
    return await resp.json();
  }
}

export const api = new ComfyApi();
