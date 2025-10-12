现在我需要整理这个代码来开源，
数据										
	18k：Q， A1，A2									
		grm								
		dpo-baseline								
		ppo-baseline								

代码			
	sft		
		grm	
		sft-baseline	
	verl		
		三个策略	
			
	eval		
		重新验证一下，eval 是要能复现的	
		其他的杂乱的结果需要清掉？	
	visualization：src_jijivski
		（）十分棘手，暂时不管
	
具体来说					
	路径相对化			
	密钥删掉				
	中文注释和大段的debug 清理				
		未用到的功能需要保留吗			
	训练的可复现性是否需要训模型验证？				
	readme 重写				
	【感觉没太大必要了】也许还需要优化代码逻辑				
					
全局清理 breakpoint()




你先分析一下， 
code 结构还是按照目前方案吗？
然后看看先把readme 写好，
然后	
    路径相对化			
	密钥删掉
这两件事可以先做起来了，你可以告诉我需要怎么做，（一些全局的查找之类的）


readme_chinese_draft:

整个项目