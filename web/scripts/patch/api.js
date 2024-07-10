// class ComfyApi extends EventTarget {
//   #registered = new Set();

//   constructor() {
//     super();
//     this.api_host = location.host;
//     this.api_base = location.pathname.split("/").slice(0, -1).join("/");
//     this.initialClientId = sessionStorage.getItem("clientId");
//   }

//   apiURL(route) {
//     if (route.startsWith('/view')) {
//       return "https://b2a6225d81e01.notebooksc.jarvislabs.net/" + route;
//     }
//     return this.api_base + route;
//   }

//   fetchApi(route, options) {
//     if (!options) {
//       options = {};
//     }
//     if (!options.headers) {
//       options.headers = {};
//     }
//     options.headers["Comfy-User"] = this.user;
//     return fetch(this.apiURL(route), options);
//   }

//   addEventListener(type, callback, options) {
//     super.addEventListener(type, callback, options);
//     this.#registered.add(type); 
//   }

//   #pollQueue() {
//     setInterval(async () => {
//       try{
//         const resp = await this.fetchApi("https://2c9ab42a769f1.notebooksb.jarvislabs.net/prompt");
//         const status = await resp.json();
//         this.dispatchEvent(new CustomEvent("status", {detail: status}));
//       } catch(error){
//         this.dispatchEvent(new CustomEvent("status", {detail: null}));
//       }
//     }, 1000);
//   }

//   createSocket(isReconnect) {
// 		if (this.socket) {
// 			return;
// 		}

// 		let opened = false;
// 		let existingSession = window.name;
// 		if (existingSession) {
// 			existingSession = "?clientId=" + existingSession;
// 		}
//     this.socket = new WebSocket("wss://b2a6225d81e01.notebooksc.jarvislabs.net//ws");
// 		this.socket.binaryType = "arraybuffer";

// 		this.socket.addEventListener("open", () => {
// 			opened = true;
// 			if (isReconnect) {
// 				this.dispatchEvent(new CustomEvent("reconnected"));
// 			}
// 		});

// 		this.socket.addEventListener("error", () => {
// 			if (this.socket) this.socket.close();
// 			if (!isReconnect && !opened) {
// 				this.#pollQueue();
// 			}
// 		});

// 		this.socket.addEventListener("close", () => {
// 			setTimeout(() => {
// 				this.socket = null;
// 				this.createSocket(true);
// 			}, 300);
// 			if (opened) {
// 				this.dispatchEvent(new CustomEvent("status", { detail: null }));
// 				this.dispatchEvent(new CustomEvent("reconnecting"));
// 			}
// 		});

// 		this.socket.addEventListener("message", (event) => {
// 			try {
//         console.log("Entered the message event listener")
// 				if (event.data instanceof ArrayBuffer) {
//           console.log("if condition of message event listener")
// 					const view = new DataView(event.data);
// 					const eventType = view.getUint32(0);
// 					const buffer = event.data.slice(4);
// 					switch (eventType) {
// 					case 1:
// 						const view2 = new DataView(event.data);
// 						const imageType = view2.getUint32(0)
// 						let imageMime
// 						switch (imageType) {
// 							case 1:
// 							default:
// 								imageMime = "image/jpeg";
// 								break;
// 							case 2:
// 								imageMime = "image/png"
// 						}
// 						const imageBlob = new Blob([buffer.slice(4)], { type: imageMime });
// 						this.dispatchEvent(new CustomEvent("b_preview", { detail: imageBlob }));
// 						break;
// 					default:
// 						throw new Error(`Unknown binary websocket message of type ${eventType}`);
// 					}
// 				}
// 				else {
//             console.log("Entered the else condition of message event listener");
// 				    const msg = JSON.parse(event.data);
// 				    switch (msg.type) {
// 					    case "status":
// 						    if (msg.data.sid) {
// 							    this.clientId = msg.data.sid;
// 							    window.name = this.clientId; // use window name so it isnt reused when duplicating tabs
// 								sessionStorage.setItem("clientId", this.clientId); // store in session storage so duplicate tab can load correct workflow
// 						    }
// 						    this.dispatchEvent(new CustomEvent("status", { detail: msg.data.status }));
// 						    break;
// 					    case "progress":
// 						    this.dispatchEvent(new CustomEvent("progress", { detail: msg.data }));
// 						    break;
// 					    case "executing":
// 						    this.dispatchEvent(new CustomEvent("executing", { detail: msg.data.node }));
// 						    break;
// 					    case "executed":
// 						    this.dispatchEvent(new CustomEvent("executed", { detail: msg.data }));
// 						    break;
// 					    case "execution_start":
// 						    this.dispatchEvent(new CustomEvent("execution_start", { detail: msg.data }));
// 						    break;
// 					    case "execution_error":
// 						    this.dispatchEvent(new CustomEvent("execution_error", { detail: msg.data }));
// 						    break;
// 					    case "execution_cached":
// 						    this.dispatchEvent(new CustomEvent("execution_cached", { detail: msg.data }));
// 						    break;
// 					    default:
// 						    if (this.#registered.has(msg.type)) {
// 							    this.dispatchEvent(new CustomEvent(msg.type, { detail: msg.data }));
// 						    } else {
// 							    throw new Error(`Unknown message type ${msg.type}`);
// 						    }
// 				    }
// 				}
// 			} catch (error) {
// 				console.warn("Unhandled message:", event.data, error);
// 			}
// 		});
// 	}

//   init() {
//     window.addEventListener(
//       "message",
//       (event) => {
//         if (event.data === "run") {
//           console.log("Run command received from parent");
//           this.performRunAction()
//             .then((data) => {
//               event.source.postMessage(
//                 {
//                   type: "result",
//                   data: data,
//                 },
//                 event.origin
//               );
//             })
//             .catch((error) => {
//               event.source.postMessage(
//                 {
//                   type: "error",
//                   message: error.message,
//                 },
//                 event.origin
//               );

//               console.error("Error during 'run' command:", error);
//             });
//         }
//       },
//       false
//     );
//   }

//   performRunAction() {
//     return new Promise((resolve, reject) => {
//       app
//         .queuePrompt()
        // .then((p) => {
        //   const json = JSON.stringify(p.output, null, 2);
        //   console.log("Generated prompt:", json);
        //   resolve({ message: json });
        // })
//         .catch((error) => {
//           console.error("Error generating prompt:", error);
//           reject(error);
//         });
//     });
//   }

//   async getExtensions() {
//     const resp = await this.fetchApi("/extensions", { cache: "no-store" });
//     return await resp.json();
//   }

//   async getEmbeddings() {
//     const resp = await this.fetchApi("/embeddings", { cache: "no-store" });
//     return await resp.json();
//   }

//   async getNodeDefs() {
//     const resp = await this.fetchApi("/object_info", { cache: "no-store" });
//     return await resp.json();
//   }

//   async queuePrompt(number, { output, workflow }) {
//     const body = {
//       client_id: this.clientId,
//       prompt: output,
//       extra_data: { extra_pnginfo: { workflow } },
//     };

//     if (number === -1) {
//       body.front = true;
//     } else if (number !== 0) {
//       body.number = number;
//     }

//     const res = await this.fetchApi("https://b2a6225d81e01.notebooksc.jarvislabs.net/prompt", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify(body),
//       mode: "no-cors",
//     });

//     if (!res.ok) {
//       const errorData = await res.json();
//       throw new Error(`Failed to queue prompt: ${errorData.message}`);
//     }

//     return res.json();
//   }

//   async getItems(type) {
// 		if (type === "queue") {
// 			return this.getQueue();
// 		}
// 		return this.getHistory();
// 	}

//   async getSystemStats() {}

//   getUserConfig = async () => ({ storage: "browser", migrated: true });

//   async getUserData(file, options) {
//     return this.fetchApi(`/userdata/${encodeURIComponent(file)}`, options);
//   }

//   async listUserData(dir, recurse, split) {
//     const resp = await this.fetchApi(
//       `/userdata?${new URLSearchParams({
//         recurse,
//         dir,
//         split,
//       })}`
//     );
//     if (resp.status === 404) return [];
//     if (resp.status !== 200) {
//       throw new Error(`Error getting user data list '${dir}': ${resp.status} ${resp.statusText}`);
//     }
//     return resp.json();
//   }

//   async getSettings() {
//     return (await this.fetchApi("/settings")).json();
//   }
// }

// export const api = new ComfyApi();


  // initWebSocket(isReconnect) {
  //   if(this.ws){
  //     return;
  //   }
  //   let opened = false;
  //   this.ws = new WebSocket("wss://2c9ab42a769f1.notebooksb.jarvislabs.net//ws");
  //   this.ws.binaryType = "arraybuffer";

  //   this.ws.addEventListener("open", () => {
  //     console.log("Web socket opened");
  //     opened = true;
  //     if(isReconnect){
  //       this.dispatchEvent(new CustomEvent("reconnected"));
  //     }
  //   })

  //   this.ws.addEventListener("error", () => {
  //     if(this.ws) {
  //       this.ws.close();
  //     }
  //     if(!isReconnect && !opened){
  //       this.#pollQueue();
  //     }
  //   })

  //   this.ws.addEventListener("close", () => {
  //     setTimeout(() => {
  //       this.ws = null;
  //       this.initWebSocket(true);
  //     }, 300);
  //     if(opened){
  //       this.dispatchEvent(new CustomEvent("status", {detail: null}))
  //       this.dispatchEvent(new CustomEvent("reconnecting"));
  //     }
  //   });

  //   this.ws.addEventListener("message", (event) => {
  //         console.log(event.data);
  //         try {
  //           if (event.data instanceof ArrayBuffer) {
  //             const view = new DataView(event.data);
  //             const eventType = view.getUint32(0);
  //             const buffer = event.data.slice(4);
  //             switch (eventType) {
  //             case 1:
  //               const view2 = new DataView(event.data);
  //               const imageType = view2.getUint32(0)
  //               let imageMime
  //               switch (imageType) {
  //                 case 1:
  //                 default:
  //                   imageMime = "image/jpeg";
  //                   break;
  //                 case 2:
  //                   imageMime = "image/png"
  //               }
  //               const imageBlob = new Blob([buffer.slice(4)], { type: imageMime });
  //               this.dispatchEvent(new CustomEvent("b_preview", { detail: imageBlob }));
  //               break;
  //             default:
  //               throw new Error(`Unknown binary websocket message of type ${eventType}`);
  //             }
  //           }
            // else {
            //     const msg = JSON.parse(event.data);
            //     switch (msg.type) {
            //       case "status":
            //         if (msg.data.sid) {
            //           this.clientId = msg.data.sid;
            //           window.name = this.clientId; // use window name so it isnt reused when duplicating tabs
            //         sessionStorage.setItem("clientId", this.clientId); // store in session storage so duplicate tab can load correct workflow
            //         }
            //         this.dispatchEvent(new CustomEvent("status", { detail: msg.data.status }));
            //         break;
            //       case "progress":
            //         this.dispatchEvent(new CustomEvent("progress", { detail: msg.data }));
            //         break;
            //       case "executing":
            //         this.dispatchEvent(new CustomEvent("executing", { detail: msg.data.node }));
            //         break;
            //       case "executed":
            //         this.dispatchEvent(new CustomEvent("executed", { detail: msg.data }));
            //         break;
            //       case "execution_start":
            //         this.dispatchEvent(new CustomEvent("execution_start", { detail: msg.data }));
            //         break;
            //       case "execution_error":
            //         this.dispatchEvent(new CustomEvent("execution_error", { detail: msg.data }));
            //         break;
            //       case "execution_cached":
            //         this.dispatchEvent(new CustomEvent("execution_cached", { detail: msg.data }));
            //         break;
            //       default:
            //         if (this.#registered.has(msg.type)) {
            //           this.dispatchEvent(new CustomEvent(msg.type, { detail: msg.data }));
            //         } else {
            //           throw new Error(`Unknown message type ${msg.type}`);
            //         }
            //     }
            // }
  //         } catch (error) {
  //           console.warn("Unhandled message:", event.data, error);
  //         }
  //   });
  // }

  // sendMessage(message) {
  //   if (this.ws && this.ws.readyState === WebSocket.OPEN) {
  //     this.ws.send(message);
  //   } else {
  //     console.error("WebSocket is not open.");
  //   }
  // }