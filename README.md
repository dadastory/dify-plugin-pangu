# pangu

**Author:** dadastory  
**Version:** 0.0.1  
**Type:** model  

## 📖 Description  

`pangu` is a **dify plugin** designed to adapt the **Pangu Large Language Model** to the OpenAI-compatible API format.  
It enables seamless integration with dify while supporting both **visible and hidden reasoning process (slow thinking)**.  

Key features:  
- ✅ Supports requests to the Pangu model.  
- ✅ Fully compatible with **OpenAI API format**, requiring minimal code changes.  
- ✅ Allows **showing or hiding the reasoning process** as needed.  
- ✅ Simple configuration — only `base_url` and `api_key` are required (same as OpenAI).  

## ⚙️ Configuration  

After installing the `pangu` plugin in dify, configure it with:  

```yaml
base_url: <your_pangu_model_base_url>
api_key: <your_pangu_model_api_key>
