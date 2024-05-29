package com.skkuse.greenserver.code;

import java.util.Base64;

public class Buggy {
    private String code;

    public void setCode(String code) {
        this.code = code;
    }

    public String base64ToJavaCode() {
        byte[] decodedBytes = Base64.getDecoder().decode(code);
        return new String(decodedBytes);
    }
}
