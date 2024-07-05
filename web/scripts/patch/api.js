class ComfyApi extends EventTarget {

    #registered = new Set();
    getUserConfig = async () => ({ storage: 'browser', migrated: true });


    constructor() {
		super();
		this.api_host = location.host;
		this.api_base = location.pathname.split('/').slice(0, -1).join('/');
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
			throw new Error(`Error getting user data list '${dir}': ${resp.status} ${resp.statusText}`);
		}
		return resp.json();
	}

    async getSystemStats() {}

    async getSettings() {
        return (await this.fetchApi("/settings")).json();
    }

    init() {}

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