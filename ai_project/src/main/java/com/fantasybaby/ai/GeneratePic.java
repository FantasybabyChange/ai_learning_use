package com.fantasybaby.ai;

import com.google.genai.Client;

public class GeneratePic {
    public static void main(String[] args) {
        // 1.创建对象
        Client client = Client.builder()
                .apiKey("AIzaSyCI98F6bRtgFdg9VBS_4T_p-u-s4zAnFL8")
                .vertexAI(true)
                .build();
        // 2.设置提示词
        Content content =
                Content.fromParts(
                        Part.fromText("将这张照片转换为一个角色手办。"),
                        Part.fromUri("\\imgs\\demo.jpg", "image/jpeg"));
        // 3.生成结果
        GenerateContentResponse response =
                client.models.generateContent("gemini-2.5-flash", content, null);
        System.out.println("Response: " + response.text());

    }
}
