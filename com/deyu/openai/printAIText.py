import openai
if __name__ == '__main__':

    # 设置 OpenAI API 密钥
    openai.api_key = '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'

    # 调用 OpenAI GPT-3 API 生成文本
    def generate_text(prompt):
        completions = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )
        message = completions.choices[0].text
        return message.strip()


    # 输入提示语，生成文本
    prompt = "解释一下背压"
    output = generate_text(prompt)

    # 输出生成的文本
    print(output)
